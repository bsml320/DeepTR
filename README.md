# DeepTR
## Prediction and characterization of T cell response by improved T cell receptors to antigen specificity with interpretable deep learning

Cellular immunity is orchestrated by T cells through their immense T-cell receptors (TCRs) repertoire, which interact with antigenic peptides presented by major histocompatibility complex (pMHC) molecules, but the specificities of the T cell response is largely undetermined because of the huge variety of TCRs. Here, we present DeepTR, a one-stop collection of unsupervised and supervised deep learning approaches for pan peptide-MHC class I binding prediction, TCR featurization, and accurate T cell response prediction. DeepTR yields higher predictive performance and more efficient feature representation for peptide binding to MHC and enables superior antigen-specific TCR featurization than current state-of-the-art approaches. Through a transfer learning strategy, DeepTR provides accurate prediction of T cell activation achieved by mimicking crucial steps of the antigen presentation pathway. DeepTR also enables the discovery of specific TCR groups with a new regulatory mechanism and characterizes important contact residues that mediate TCR-antigen binding specificity. DeepTR may advance our understanding of the mechanisms of T cell-mediated immunity and yield new insight in both personalized immune treatment and development of targeted vaccines. DeepTR is freely available at https://bioinfo.uth.edu/DeepTR.

<div align=center><img src="https://bioinfo.uth.edu/DeepTR/images/Picture1.png" width="800px"></div>

# Installation
Download DeepTR by
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

# Usage
Please cd to the folder which contains DeepTR_Prediction.py.
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
