# Elucidating tumor heterogeneity from spatially resolved transcriptomics data by multi-view graph collaborative learning.

![image](https://github.com/cmzuo11/stMVC/blob/main/Utilities/Main_figure_stMVC.png)

Overview of stMVC model. a Given each spatially resolved transcriptomics (SRT) data data with four-layer profiles: histological images (I), spatial locations (S), gene expression (X), and manual cell segmentation (Y) as the input, stMVC integrated them to disentangle tissue heterogeneity, particularly for the tumor. b stMVC adopted SimCLR model with feature extraction framework from ResNet-50 to efficiently learn visual features (h_i) for each spot (v_i) by maximizing agreement between differently augmented views of the same spot image (I_i) via a contrastive loss in the latent space (l_i), and then constructed HSG by the learned visual features h_i. c stMVC model adopting a SGATE learned view-specific representations (〖P_i〗^1 and 〖P_i〗^2) for each of two graphs including HSG and SLG, as well as the latent feature from gene expression data by the autoencoder-based framework as a feature matrix, where a SGATE for each view was trained under weak supervision of the cell segmentation to capture its efficient low-dimensional manifold structure, and simultaneously integrated two-view graphs for robust representations (R_i) by learning weights of different views via attention mechanism. d Robust representations R_i can be used for elucidating tumor heterogeneity: detecting spatial domains, visualizing the relationship distance between different domains, and further denoising data.

# Installation

## Install stMVC

Installation Tested on Red Hat 7.6 with Python 3.6.12 and torch 1.6.0 on a machine with one 40-core Intel(R) Xeon(R) Gold 5115 CPU addressing with 132GB RAM, and two NVIDIA TITAN V GPU addressing 24GB. stMVC is implemented in the Pytorch framework. Please run stMVC on CUDA if possible. 

### grabbing source code

```
git clone https://github.com/cmzuo11/stMVC.git

cd stMVC
```

### install stMVC in the virutal environment by conda

The used packages (described by "used_package.txt") for stMVC can be automatically installed.

```
conda create -n stMVC python=3.6.12 pip

source activate

conda activate stMVC

pip install -r used_package.txt
```

## Install histological label software (labelme) 

Installation tested on Windows 10 with Intel Core i7-4790 CPU, and the labelme software is available at Github: https://github.com/wkentaro/labelme

# Quick start

## Input: 

* a general output of 10X pipeline, for example, a directory includes a file named as filtered_feature_bc_matrix.h5, a directory named as spatial with at least four files: tissue_positions_list.csv, tissue_hires_image.png, metrics_summary_csv.csv, scalefactors_json.json, and a directory named as filtered_feature_bc_matrix with three files: matrix.mtx.gz, features.tsv.gz, and barcodes.tsv.gz;  

* with the example file of slice 151673 as an example, you can download it by the following code:
```
wget https://zenodo.org/record/5977605/files/stMVC_test_data.zip

unzip stMVC_test_data.zip
```
## Run: 

* python main_stMVC_DLPFC.py

## Useful paramters:

* modify the initial learning rate paramters for learning view-specific representations by single-view graph and robust representations by multi-view graph. i.e., lr_T1 for HSG, lr_T2 for SLG, lr_T3 for collaborative learning. The default value of three parameters is 0.002. You can adjust them from 0.001 to 0.003 by 0.001;

## Output:

## the output file will be saved for further analysis:

* GAT_2-view_model.pth: saved model for reproducing results.

* GAT_2-view_robust_representation.csv: robust representations for latter clustering, visualization, and data denoising.

# Reference

Chunman Zuo, Yijian Zhang, Chen Cao, Jinwang Feng, Mingqi Jiao, and Luonan Chen. Elucidating tumor heterogeneity from spatially resolved transcriptomics data by multi-view graph collaborative learning. 2022. (submitted).
