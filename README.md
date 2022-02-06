# Elucidating tumor heterogeneity from spatially resolved transcriptomics data by multi-view graph collaborative learning.

![image](https://github.com/cmzuo11/stMVC/blob/main/Utilities/Main_figure_stMVC.png)

Overview of stMVC model. a Given each spatially resolved transcriptomics (SRT) data data with four-layer profiles: histological images (I), spatial locations (S), gene expression (X), and manual cell segmentation (Y) as the input, stMVC integrated them to disentangle tissue heterogeneity, particularly for the tumor. b stMVC adopted SimCLR model with feature extraction framework from ResNet-50 to efficiently learn visual features (h_i) for each spot (v_i) by maximizing agreement between differently augmented views of the same spot image (I_i) via a contrastive loss in the latent space (l_i), and then constructed HSG by the learned visual features h_i. c stMVC model adopting a SGATE learned view-specific representations (〖P_i〗^1 and 〖P_i〗^2) for each of two graphs including HSG and SLG, as well as the latent feature from gene expression data by the autoencoder-based framework as a feature matrix, where a SGATE for each view was trained under weak supervision of the cell segmentation to capture its efficient low-dimensional manifold structure, and simultaneously integrated two-view graphs for robust representations (R_i) by learning weights of different views via attention mechanism. d Robust representations R_i can be used for elucidating tumor heterogeneity: detecting spatial domains, visualizing the relationship distance between different domains, and further denoising data.

# Installation

## Install stMVC

Installation was tested on Red Hat 7.6 with Python 3.6.12 and torch 1.6.0 on a machine with one 40-core Intel(R) Xeon(R) Gold 5115 CPU addressing with 132GB RAM, and two NVIDIA TITAN V GPU addressing 24GB. stMVC is implemented in the Pytorch framework. Please run stMVC on CUDA if possible. 

#### 1. grab source code of stMVC

```
git clone https://github.com/cmzuo11/stMVC.git

cd stMVC
```

#### 2. install stMVC in the virtual environment by conda 

* Firstly, install conda: https://docs.anaconda.com/anaconda/install/index.html

* Then, automatically install all used packages (described by "used_package.txt") for stMVC in a few mins.

```
conda create -n stMVC python=3.6.12 pip

source activate

conda activate stMVC

pip install -r used_package.txt
```

## Install histological image annotation software (labelme) 

Installation was tested on Windows 10 with Intel Core i7-4790 CPU, and the labelme software is available at Github: https://github.com/wkentaro/labelme

## Install R packages 

* install tested on R =4.0.0

* install package 'Seurat' based on the Github https://github.com/satijalab/seurat

# Quick start

## Input: 

* a general output of 10X pipeline, for example, a directory includes a file named filtered_feature_bc_matrix.h5, a directory named spatial with five files: tissue_positions_list.csv, tissue_lowres_image.png, tissue_hires_image.png, metrics_summary_csv.csv, scalefactors_json.json, and a directory named as filtered_feature_bc_matrix with three files: matrix.mtx.gz, features.tsv.gz, and barcodes.tsv.gz;  

* take slice 151673 as an example, you can download it by the following scripts:

```
wget https://zenodo.org/record/5977605/files/stMVC_test_data.zip

unzip stMVC_test_data.zip
```

Note: the folder named 'DLPFC_151673' is the raw data of slice 151673.

## Run: 

### step 1. Preprocess raw data

This function automatically (1) learns 50-dimensional features from 2000 highly variable genes of gene expression data, (2) trains SimCLR model (500 iterations) by data augmentations and contrastive learning and extracts 2048-dimensional visual features from histological data, and (3) saves the physical location of each spot into a file 'Spot_location.csv' into a folder named spatial of the current directory.

```
python Preprcessing_stMVC.py --basePath ./stMVC_test_data/DLPFC_151673/ 
```

The running time mainly depends on the iteration of SimCLR training. It takes 3.7h to generate the above-described files. You can modify the following parameters to reduce time:

* batch_size_I: defines the batch size for training SimCLR model. The default value is 128. You can modify it based on your memory size. The larger the parameter, the less time.

* max_epoch_I: defines the max iteration for training SimCLR model. The default value is 500. You can modify it. The smaller the parameter, the less time.

To reproduce the result, you should use the default parameters.

Note: To reduce the waiting time, we have uploaded our preprocessed data into the ./stMVC_test_data/DLPFC_151673/stMVC/ folder. You can directly perform step3.

### step 2. Manual cell segmentation (for OEAD and IDC dataset)

This function defines the classification of each spot based on our manual cell segmentation by labelme software, and saves the cell segmentation file (Image_cell_segmentation_0.5.csv) into the 'image_segmentation' directory. It takes ~ XX mins.

* run lableme software, manual outline each tumor region based on our defined strategy, and save the annotation into a json file named 'tissue_hires_image.json'.

* defines the classification for each spot based on above-generated json file. Here, we used IDC dataset as an example.

```
python Image_cell_segmentation.py --basePath ./stMVC_test_data/IDC/ --jsonFile tissue_hires_image.json
```
Note: To reduce the waiting time, we have uploaded the tissue_hires_image.json and the processed result from step 1 into a folder named IDC. You can directly perform step3.

### step 3. Run stMVC model

This function automatically learns robust representations by multi-view graph collaborative learning. It takes ~7 min.

```
python stMVC_model.py --basePath ./stMVC_test_data/DLPFC_151673/ 
```
In running, the useful parameters:

* lr_T1 for HSG, lr_T2 for SLG, lr_T3 for collaborative learning: defines learning rate parameters for learning view-specific representations by single-view graph and robust representations by multi-view graph. i.e., . The default value of three parameters is 0.002. You can adjust them from 0.001 to 0.003 by 0.001;

* max_epoch_T: defines the max iteration for training view-specific graph or multi-view graph. The default value is 500. You can modify it. The larger the parameter, the more time.

* beta_pa: defines the penaly for the knowledge transfer from robust representations to view-specific representations. The default value is 8.

* knn: defines the K-nearest similarity spots for each spot to construct HSG or SLG. The default value is 7 where the K-nearest spots includes itself. 

To reproduce the result, you should use the default parameters.

## Output:

## the output file will be saved for further analysis:

* GAT_2-view_model.pth: a saved model for reproducing results.

* GAT_2-view_robust_representation.csv: robust representations for latter clustering, visualization, and data denoising.

# References

* MVE: https://github.com/mnqu/MVE
* GAT: https://github.com/gordicaleksa/pytorch-GAT
* SimCLR: https://github.com/google-research/simclr
* stLearn: https://github.com/BiomedicalMachineLearning/stLearn
* DCCA: https://github.com/cmzuo11/DCCA

# Citation

Chunman Zuo, Yijian Zhang, Chen Cao, Jinwang Feng, Mingqi Jiao, and Luonan Chen. Elucidating tumor heterogeneity from spatially resolved transcriptomics data by multi-view graph collaborative learning. 2022. (submitted).
