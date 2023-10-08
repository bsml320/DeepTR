# %%
import os
import re
import sys
import copy
import random
import numpy as np
import pandas as pd
import argparse
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import concatenate 
from keras import optimizers
from allennlp.commands.elmo import ElmoEmbedder
import torch
from keras.models import model_from_json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %%
common_aa = "ARNDCQEGHILKMFPSTWYV"
aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}     
path_dict = './supporting_file/'
model_dict = './models/'
pep_length = [8,9,10,11,12,13]

cdrs = np.load(path_dict + "create_cdr_dict.npy", allow_pickle = True).item()

cdrs1 = copy.copy(cdrs)
for key in list(cdrs1['human'].keys()):
    dict_q = cdrs1['human'][key]
    for key1 in list(dict_q.keys()):
        key2 = key1.split('*')[0]
        if key2 not in cdrs1['human'][key].keys(): 
            cdrs['human'][key][key2] = dict_q[key1]

cdrs2 = copy.copy(cdrs)
for key in list(cdrs2['human'].keys()):
    dict_q = cdrs2['human'][key]
    for key1 in list(dict_q.keys()):
        key2 = key1.replace('-','.')
        if key2 not in cdrs2['human'][key].keys(): 
            cdrs['human'][key][key2] = dict_q[key1]

cdrs3 = copy.copy(cdrs)
for key in list(cdrs3['human'].keys()):
    dict_q = cdrs3['human'][key]
    for key1 in list(dict_q.keys()):
        key2 = key1.split('*')[0].replace('/','')
        if key2 not in cdrs3['human'][key].keys(): 
            cdrs['human'][key][key2] = dict_q[key1]
            
def read_blosum(path,one_hot):
    f = open(path,"r")
    blosum = []
    if one_hot ==0: #(blosum 50)
       for line in f:
           blosum.append([(float(i))/10 for i in re.split("\t",line)])
    else:
        for line in f: #(one-hot)
           blosum.append([float(i) for i in re.split("\t",line)])
        #The values are rescaled by a factor of 1/10 to facilitate training
    f.close()
    return blosum

def pseudo_seq(seq_dict,blosum_matrix):
    pseq_dict = {}#pseudo sequence dictionary
    for allele in seq_dict.keys():
        new_pseq = []
        for index in range(34):
            new_pseq.append(blosum_matrix[aa[seq_dict[allele][index]]]) 
        pseq_dict[allele] = new_pseq
    return pseq_dict

blosum_matrix = read_blosum(path_dict + 'blosum50.txt', 0)
pseq_dict = np.load(path_dict + 'pseq_dict_all.npy', allow_pickle = True).item()
pseq_dict_blosum_matrix = pseudo_seq(pseq_dict, blosum_matrix)

def peptide_matrix(peptides):
    
    blosum_matrix = np.array([[0.5,  -0.2,  -0.1,  -0.2,  -0.1,  -0.1,  -0.1,  0.0,  -0.2,  -0.1,  -0.2,  -0.1,  -0.1,  -0.3,  -0.1,  0.1,  0.0,  -0.3,  -0.2,  0.0], 
    [-0.2,  0.7,  -0.1,  -0.2,  -0.4,  0.1,  0.0,  -0.3,  0.0,  -0.4,  -0.3,  0.3,  -0.2,  -0.3,  -0.3,  -0.1,  -0.1,  -0.3,  -0.1,  -0.3], 
    [-0.1,  -0.1,  0.7,  0.2,  -0.2,  0.0,  0.0,  0.0,  0.1,  -0.3,  -0.4,  0.0,  -0.2,  -0.4,  -0.2,  0.1,  0.0,  -0.4,  -0.2,  -0.3], 
    [-0.2,  -0.2,  0.2,  0.8,  -0.4,  0.0,  0.2,  -0.1,  -0.1,  -0.4,  -0.4,  -0.1,  -0.4,  -0.5,  -0.1,  0.0,  -0.1,  -0.5,  -0.3,  -0.4],
    [-0.1,  -0.4,  -0.2,  -0.4,  1.3,  -0.3,  -0.3,  -0.3,  -0.3,  -0.2,  -0.2,  -0.3,  -0.2,  -0.2,  -0.4,  -0.1,  -0.1,  -0.5,  -0.3,  -0.1],
    [-0.1,  0.1,  0.0,  0.0,  -0.3,  0.7,  0.2,  -0.2,  0.1,  -0.3,  -0.2,  0.2,  0.0,  -0.4,  -0.1,  0.0,  -0.1,  -0.1,  -0.1,  -0.3],
    [-0.1,  0.0,  0.0,  0.2,  -0.3,  0.2,  0.6,  -0.3,  0.0,  -0.4,  -0.3,  0.1,  -0.2,  -0.3,  -0.1,  -0.1,  -0.1,  -0.3,  -0.2,  -0.3],
    [0.0,  -0.3,  0.0,  -0.1,  -0.3,  -0.2,  -0.3,  0.8,  -0.2,  -0.4,  -0.4,  -0.2,  -0.3,  -0.4,  -0.2,  0.0,  -0.2,  -0.3,  -0.3,  -0.4],
    [-0.2,  0.0,  0.1,  -0.1,  -0.3,  0.1,  0.0,  -0.2,  1.0,  -0.4,  -0.3,  0.0,  -0.1,  -0.1,  -0.2,  -0.1,  -0.2,  -0.3,  0.2,  -0.4],
    [-0.1,  -0.4,  -0.3,  -0.4,  -0.2,  -0.3,  -0.4,  -0.4,  -0.4,  0.5,  0.2,  -0.3,  0.2,  0.0,  -0.3,  -0.3,  -0.1,  -0.3,  -0.1,  0.4],
    [-0.2,  -0.3,  -0.4,  -0.4,  -0.2,  -0.2,  -0.3,  -0.4,  -0.3,  0.2,  0.5,  -0.3,  0.3,  0.1,  -0.4,  -0.3,  -0.1,  -0.2,  -0.1,  0.1],
    [-0.1,  0.3,  0.0,  -0.1,  -0.3,  0.2,  0.1,  -0.2,  0.0,  -0.3,  -0.3,  0.6,  -0.2,  -0.4,  -0.1,  0.0,  -0.1,  -0.3,  -0.2,  -0.3],
    [-0.1,  -0.2,  -0.2,  -0.4,  -0.2,  0.0,  -0.2,  -0.3,  -0.1,  0.2,  0.3,  -0.2,  0.7,  0.0,  -0.3,  -0.2,  -0.1,  -0.1,  0.0,  0.1],
    [-0.3,  -0.3,  -0.4,  -0.5,  -0.2,  -0.4,  -0.3,  -0.4,  -0.1,  0.0,  0.1,  -0.4,  0.0,  0.8,  -0.4,  -0.3,  -0.2,  0.1,  0.4,  -0.1],
    [-0.1,  -0.3,  -0.2,  -0.1,  -0.4,  -0.1,  -0.1,  -0.2,  -0.2,  -0.3,  -0.4,  -0.1,  -0.3,  -0.4,  1.0,  -0.1,  -0.1,  -0.4,  -0.3,  -0.3],
    [0.1,  -0.1,  0.1,  0.0,  -0.1,  0.0,  -0.1,  0.0,  -0.1,  -0.3,  -0.3,  0.0,  -0.2,  -0.3,  -0.1,  0.5,  0.2,  -0.4,  -0.2,  -0.2],
    [0.0,  -0.1,  0.0,  -0.1,  -0.1,  -0.1,  -0.1,  -0.2,  -0.2,  -0.1,  -0.1,  -0.1,  -0.1,  -0.2,  -0.1,  0.2,  0.5,  -0.3,  -0.2,  0.0],
    [-0.3,  -0.3,  -0.4,  -0.5,  -0.5,  -0.1,  -0.3,  -0.3,  -0.3,  -0.3,  -0.2,  -0.3,  -0.1,  0.1,  -0.4,  -0.4,  -0.3,  1.5,  0.2,  -0.3],
    [-0.2,  -0.1,  -0.2,  -0.3,  -0.3,  -0.1,  -0.2,  -0.3,  0.2,  -0.1,  -0.1,  -0.2,  0.0,  0.4,  -0.3,  -0.2,  -0.2,  0.2,  0.8,  -0.1],
    [0.0,  -0.3,  -0.3,  -0.4,  -0.1,  -0.3,  -0.3,  -0.4,  -0.4,  0.4,  0.1,  -0.3,  0.1,  -0.1,  -0.3,  -0.2,  0.0,  -0.3,  -0.1,  0.5]])
    
    aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}
    pep_length = [8,9,10,11,12,13]
    
    if set(list(peptides)).difference(list('ACDEFGHIKLMNPQRSTVWY')):
        return   
    if len(peptides) not in pep_length:
        return 
    pep_blosum = []
    for residue_index in range(13):
        if residue_index < len(peptides):
            pep_blosum.append(blosum_matrix[aa[peptides[residue_index]]])
        else:
            pep_blosum.append(np.zeros(20))
    for residue_index in range(13):
        if 13 - residue_index > len(peptides):
            pep_blosum.append(np.zeros(20)) 
        else:
            pep_blosum.append(blosum_matrix[aa[peptides[len(peptides) - 13 + residue_index]]])
            
    return pep_blosum    

# %%
def import_binding_model(model_dict, number):
    models = []
    for i in range(number):
        json_f = open(model_dict + "binding/model_"+str(i)+".json", 'r')
        loaded_model_json = json_f.read()
        json_f.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights((model_dict + "binding/model_"+str(i)+".h5"))
        models.append(loaded_model)  
    return models

def import_prediction_model(model_dict, number):
    models = []
    for i in range(number):
        json_f = open(model_dict + "dnn/model_dnn.json", 'r')
        loaded_model_json = json_f.read()
        json_f.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights((model_dict + "dnn/model_"+str(i)+".h5"))
        models.append(loaded_model)  
    return models

def scoring(models, data):
    import numpy as np
    probas_ = [np.transpose(model.predict(data))[0] for model in models]
    probas_ = [np.mean(scores) for scores in zip(*probas_)]
    return probas_  

def embedding_model(model_dict):
    json_f = open(model_dict + 'pan_binding/model_pan_binding.json', 'r')
    loaded_model_json = json_f.read()
    json_f.close()
    pan_ligand = model_from_json(loaded_model_json)
    pan_ligand.load_weights(model_dict + 'pan_binding/model_pan_binding.h5') 
    layer_model = Model(inputs=pan_ligand.input, outputs=pan_ligand.layers[26].output)
    return layer_model

weights = model_dict + 'uniref50_v2/weights.hdf5'
options = model_dict + 'uniref50_v2/options.json'
seqvec  = ElmoEmbedder(options,weights,cuda_device=-1)

model_embedding = embedding_model(model_dict)
model_binding = import_binding_model(model_dict, 5)
model_prediction = import_prediction_model(model_dict, 5)

json_file = open(model_dict + 'lstm/LSTM_Audoencoder_8.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
modelx = model_from_json(loaded_model_json)
modelx.load_weights(model_dict + "lstm/LSTM_Audoencoder_8.h5")

def s2v(seq):
    embed1 = seqvec.embed_sentence(list(seq)) 
    protein_embd1 = torch.tensor(embed1).sum(dim=0).mean(dim=0) 
    return list(protein_embd1.detach().numpy())

def pred_and_write_metrics_datatable(TCR_file, tmp_folder):
    df_tcrs = pd.read_csv(TCR_file)
    TCRs = df_tcrs.values.tolist()
    
    print("Loading TCRs information for prediction...")
    
    ac1 = []
    ac2 = []
    ac2_5 = []
    bc1 = []
    bc2 = []
    bc2_5 = []
    ac3 = []
    bc3 = []
    epi = []
    hla = []

    L31 = []
    for out in TCRs:
        try:
            ac3.append(out[0])
            bc3.append(out[2])
            TRAV = cdrs['human']['A'][out[1]] 
            ac1.append(TRAV[0].replace('-',''))
            ac2.append(TRAV[1].replace('-',''))
            ac2_5.append(TRAV[2].replace('-',''))
            TRAB = cdrs['human']['B'][out[3]] 
            bc1.append(TRAB[0].replace('-',''))
            bc2.append(TRAB[1].replace('-',''))
            bc2_5.append(TRAB[2].replace('-',''))
            epi.append(out[-2])
            hla.append(out[-1])
        except Exception as e:
            print(e)
            continue
        L31.append(out)

    # %%
    embed_ac1 = dict(zip(list(set(ac1)), map(lambda x: s2v(x), list(set(ac1)))))
    embed_ac2 = dict(zip(list(set(ac2)), map(lambda x: s2v(x), list(set(ac2)))))
    embed_ac2_5 = dict(zip(list(set(ac2_5)), map(lambda x: s2v(x), list(set(ac2_5)))))
    embed_ac3 = dict(zip(list(set(ac3)), map(lambda x: s2v(x), list(set(ac3)))))

    embed_bc1 = dict(zip(list(set(bc1)), map(lambda x: s2v(x), list(set(bc1)))))
    embed_bc2 = dict(zip(list(set(bc2)), map(lambda x: s2v(x), list(set(bc2)))))
    embed_bc2_5 = dict(zip(list(set(bc2_5)), map(lambda x: s2v(x), list(set(bc2_5)))))
    embed_bc3 = dict(zip(list(set(bc3)), map(lambda x: s2v(x), list(set(bc3)))))

    embed_epi = dict(zip(list(set(epi)), map(lambda x: peptide_matrix(x), list(set(epi)))))
    embed_hla = dict(zip(list(set(hla)), map(lambda x: pseq_dict_blosum_matrix[x], list(set(hla)))))

    print("TCRs characterization is in progress...")
    
    # %%
    test_tcr, test_pmhc = [], []
    for i,x in enumerate(L31):
        test_tcr.append([embed_ac1[ac1[i]], embed_ac2[ac2[i]], embed_ac2_5[ac2_5[i]],
        embed_ac3[ac3[i]], embed_bc1[bc1[i]], embed_bc2[bc2[i]], embed_bc2_5[bc2_5[i]], 
        embed_bc3[bc3[i]]]) 
        
        test_pmhc.append([embed_epi[epi[i]], embed_hla[hla[i]]])

    test_data = np.array(test_tcr)  
    test_tcr1 = modelx.predict(test_data) 
    test_pmhc1 = np.array(test_pmhc)

    [validation_pep, validation_mhc] = [[i[j] for i in test_pmhc1] for j in range(2)] 
    mp_pairs1 =  [np.array(validation_pep),np.array(validation_mhc)]  

    layer_output_array = model_embedding.predict(mp_pairs1)
    binding_scores = scoring(model_binding, mp_pairs1)

    X_test = np.concatenate((test_tcr1, layer_output_array), axis=1)
    interaction_scores = scoring(model_prediction, X_test)

    # %%
    binding_predictions: pd.DataFrame = pd.DataFrame(columns=['CDR3a', 'CDR3a_V', 'CDR3b',
                                'CDR3b_V', 'Antigens','HLA_alleles',
                                'Score_BA','Score_interaction'])
    for i, line in enumerate(L31): 
        rows = []
        rows.append(line + [binding_scores[i], interaction_scores[i]])
        binding_predictions = binding_predictions.append(
            pd.DataFrame(columns=['CDR3a', 'CDR3a_V', 'CDR3b',
                                    'CDR3b_V', 'Antigens','HLA_alleles',
                                'Score_BA','Score_interaction'], data=rows), ignore_index=True)    

    # tmp_folder = './prediction'
    if not os.path.isdir(tmp_folder):
        os.makedirs(tmp_folder) 
        
    binding_predictions.to_csv(tmp_folder + '/DeepTR_predictions.tsv', index=False, header=True)  

    print("Done! the prediction results are written to file, please check...")
    
def main(args): 
    tcr_file = args.inputfile
    out_file = args.outfile
    pred_and_write_metrics_datatable(tcr_file, out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prediction of T cell response by improved T cell receptors to antigen specificity")
    parser.add_argument('-inf', '--inputfile', type=str,
                        help='One file containing predicted sequences.') 
    parser.add_argument('-out', '--outfile', type=str, 
                    help='The output of the predicted result.') 
                     
    args = parser.parse_args()
    main(args)
    # python DeepTR_Prediction.py -inf './data/test_tcr.txt' -out './prediction'