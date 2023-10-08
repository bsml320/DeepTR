# DeepTR
## Prediction and characterization of T cell response by improved T cell receptors to antigen specificity with interpretable deep learning

Cellular immunity is orchestrated by T cells through their immense T-cell receptors (TCRs) repertoire, which interact with antigenic peptides presented by major histocompatibility complex (pMHC) molecules, but the specificities of the T cell response is largely undetermined because of the huge variety of TCRs. Here, we present DeepTR, a one-stop collection of unsupervised and supervised deep learning approaches for pan peptide-MHC class I binding prediction, TCR featurization, and accurate T cell response prediction. DeepTR yields higher predictive performance and more efficient feature representation for peptide binding to MHC and enables superior antigen-specific TCR featurization than current state-of-the-art approaches. Through a transfer learning strategy, DeepTR provides accurate prediction of T cell activation achieved by mimicking crucial steps of the antigen presentation pathway. DeepTR also enables the discovery of specific TCR groups with a new regulatory mechanism and characterizes important contact residues that mediate TCR-antigen binding specificity. DeepTR may advance our understanding of the mechanisms of T cell-mediated immunity and yield new insight in both personalized immune treatment and development of targeted vaccines. DeepTR is freely available at https://bioinfo.uth.edu/DeepTR.

<div align=center><img src="https://bioinfo.uth.edu/DeepTR/images/Picture1.png" width="800px"></div>

# Installation
Download NetRNApan by
```
git clone https://github.com/BioDataStudy/DeepTR.git
```
Installation has been tested in Linux server, CentOS Linux release 7.8.2003 (Core), with Python 3.7. Since the package is written in python 3x, python3x with the pip tool must be installed. DeepTR uses the following dependencies: numpy, scipy, pandas, h5py, torch, allennlp, keras version=2.3.1, tensorflow=1.15 shutil, and pathlib. We highly recommend that users leave a message under the DeepTR issue interface (https://github.com/BioDataStudy/DeepTR/issue) when encountering any installation and running problems. We will deal with it in time. You can install these packages by the following commands:
```
conda create -n DeepTR python=3.7
conda activate DeepTR
pip install pandas
pip install numpy
pip install scipy
pip install torch
pip install allennlp==0.9.0
pip install -v keras==2.3.1
pip install -v tensorflow==1.15
pip install seaborn
pip install shutil
pip install protobuf==3.20
pip install h5py==2.10.0
```
# Performance
## pMHC module
To access the prediction performance of Net-pMHC, a five-fold cross-validation was performed on the curated binding affinity dataset. The receiver operating characteristic (ROC) curves were drawn and the corresponding AUC values were calculated. As a result, when applying this hybrid deep learning architecture, the Net-pMHC, which automatically learns discriminative features and essential residues from the peptides along the layer hierarchy, performed well with the average AUC values of 0.950 from five-fold CV. In addition, Net-pMHC also achieved a high AUC value of 0.942 when using the independent test dataset, indicating the adaptability of our model. We also calculated PCC between predicted and quantitative binding affinities to further evaluate the predictive ability of our model. Net-pMHC achieved PCCs of 0.88 and 0.84 in five-fold CV and independent testing, respectively, a good improvement when compared to the previously reported netMHCpan (BA option) with PCC as 0.76. As above, Net-pMHC was accurate and robust for the peptide-MHC binding prediction, in light of consistent and promising AUC and PCCs in both five-fold CV and independent testing. Although the output layer is dedicated to predicting antigen binding to MHC, the layers before it contain important information regarding the overall structure of the pMHC complex. Thus, we visualized the pMHCs for several well characterized MHC alleles, e.g., HLA-A*02:01, using UMAP method based on the feature representation generated from different network layers. We found our model could hierarchically learn a more efficient and interpretive feature representation of the pMHCs. More specifically, the predicted features for pMHCs and non-pMHCs were mixed at the input layer. However, as the predicted features passed through the CNN and LSTM layers, the model began differentiating between pMHCs and non-pMHCs. The attention layer could distribute higher weights to the essential positions for pMHCs prediction. When integrating with the output from the LSTM layer, pMHCs and non-pMHCs tended to separated clearly, indicating this method could efficiently infer feature representation. Overall, we demonstrated that the Net-pMHC could provide accurate pan peptide-MHC class I binding prediction and generate representative immediate numeric embedding of binding preference for pMHCs.

<div align=center><img src="https://bioinfo.uth.edu/DeepTR/images/Figure%202.jpg" width="800px"></div>

# Interpretability
The accuracy and robustness of the NetRNApan might be partly attributed to its deep neural network architecture, which is easily interpretable compared to the traditional machine-learning algorithms. The inputs can be projected via the hidden layers of NetRNApan to a representation space with lower dimensions. We used the UMAP approach to visually display the m5U sites and non-m5U sites in the training dataset based on the feature learnt at various network layers to demonstrate the capabilities of hierarchical representation using NetRNApan. We found that the feature representation became more discriminative along the network layer hierarchy. More specifically, the feature representations for m5U sites and non-m5U sites were mixed at the input layer. As the model continued to train, all nucleotides were grouped into two distinct clusters by the low-dimensional projection, reflecting binding specificities between m5U sites and non-m5U sites. 

<div align=center><img src="https://bioinfo.uth.edu/iapp/github/NetRNApan/Figure 3.jpg" width="800px"></div>

# Motif discovery
To interpret our deep learning model, we also decoded and analyzed the sequence features captured by our model from input nucleotides. Briefly, we first obtained the local segments captured by 256 convolution filters in the first convolution layer of our model. Each filter with a length of 10 nucleotides that was maximally activated at different regions of input. These activated features were then overlaid together to create position weight matrices (PWMs), which were considered as local motifs. To determine the representative motifs, motif score was calculated to measure the difference in the mean maximum activation scores between positive class and negative class. In total, 135 informative motifs were identified. Some strong motifs with higher scores were significantly enriched in positive samples such as “xCxGGG[A/U]x[C/U]U” (Kernel 101, score = 0.446, p-value < 0.01), “[C/U]xGxxxxGCG” (Kernel 138, score = 0.379, p-value < 0.01), and “UUCGAxxCxG” (Kernel 195, score = 0.365, p-value < 0.01). All motifs and the corresponding PWMs were graphically illustrated. Due to the significance of some motifs in modification, they could be found more than once. Then, the top 50 motifs were subjected to clustering analysis for further pattern mining. The pairwise Spearman's correlations between PWM of each motif were evaluated and hierarchical clustering was performed to the correlation matrix, yielding five representative motif clusters, such as UUCGAx[U/C], [C/G]GGUU[C/U]xAA, GGxCCCGG. We further analyzed one of the top-scoring motifs (Kernel 101, score = 0.446), and displayed the activation positions and distribution of the activation scores between m5U and non-m5U nucleotides. It was found that the motif had significantly greater activation scores and was enriched in modification sites and surrounding locations. Taken together, NetRNApan could identify particular patterns of recognition and revealed consensus motifs that may be significant for RNA modifications

<div align=center><img src="https://bioinfo.uth.edu/iapp/github/NetRNApan/Figure 5.jpg" width="800px"></div>

# Post-transcriptional regulation
Compared with the well characterized m6A modification and its increasingly prominent regulatory mechanism, there are still many unknown types of RNA modification. NetRNApan can provide a protein-binding perspective for understanding the functions of RNA modifications. In this study, we discovered 21 RBPs whose binding sites were significantly linked to the extracted motifs among the top 50 top-scoring motifs. For example, among all the identified RBPs, we found that the motif-43 identified by NetRNApan could be matched to the binding motif of ANKHD1 (p-value: 4.68E-03), which was found to interact with the major m5U methyltransferase TRMT2A. ANKHD1 is a large protein characterized by the presence of multiple ankyrin repeats and a K-homology (KH) domain, and its KH domain binds to RNA or ssDNA and is associated with transcriptional and translational regulation. In addition, functional enrichment analysis of these identified RBPs was also performed to further explore the potential regulatory roles of m5U. We found that several terms, including G-quadruplex RNA binding, regulation of (alternative) mRNA splicing, RNA transport and gene expression were enriched, which implied for their potential critical roles in transcriptional regulation.

<div align=center><img src="https://bioinfo.uth.edu/iapp/github/NetRNApan/Figure 6.jpg" width="800px"></div>

# Usage
Please cd to the NetBCE/prediction/ folder which contains predict.py.
Example: 
```
cd NetBCE/prediction/
python NetBCE_prediction.py -f ../testdata/test.fasta -o ../result/test_result
```
For details of other parameters, run:
```
python NetBCE_prediction.py --help
```
# Citation
Please cite the following paper for using: Xu H, Zhao Z. Deciphering RNA modification and post-transcriptional regulation by deep learning framework. In submission.
