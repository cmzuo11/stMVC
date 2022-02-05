# Elucidating tumor heterogeneity from spatially resolved transcriptomics data by multi-view graph collaborative learning.

![image](https://github.com/cmzuo11/stMVC/blob/main/Utilities/Main_figure_stMVC.png)

Overview of stMVC model. (A) Given each SRT data with four-layer profiles: histological images (I), spatial locations (S), gene expression (X), and manual cell segmentation (Y) as the input, stMVC integrated them to disentangle tissue heterogeneity, particularly for the tumor. (B) The stMVC adopted SimCLR model with feature extraction model from ResNet-50 to efficiently learn visual feature (h_i) for each spot (v_i) by maximizing agreement between differently augmented views of the same spot image (I_i) via a contrastive loss in the latent space (l_i), and then constructed HSG by the learned visual feature h_i. (C) The stMVC model adoptedapplied a semi-supervised graph attention autoencoder (SGATE) to learn view-specific representation (〖P_i〗^1 and 〖P_i〗^2) for each of two graphs including HSG and SLG, where an SGATE for each view was trained under weak supervision of the biological contexts to capture its efficient low-dimensional manifold structure, and simultaneously integrated two-view graphs for robust representation (R_i) by learning weights of different views via attention mechanism. (D) The robust representation R_i can be used for elucidating tumor heterogeneity: detecting spatial domains, visualizing the relationship distance between different domains, and further denoising data.

# Installation

stMVC is implemented in the Pytorch framework. Please run stMVC on CUDA if possible. stMVC requires python 3.6.12, and torch 1.6.0. The used packages (described by "used_package.txt") for stMVC can be automatically installed.

* git clone https://github.com/cmzuo11/stMVC.git

* cd stMVC

* python setup.py install
