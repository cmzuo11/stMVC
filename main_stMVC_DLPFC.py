# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:16:25 2021

@author: chunman zuo
"""
#import stlearn as st

import numpy as np
import pandas as pd
import os
import torch
import random
from pathlib import Path

from stMVC.utilities import parameter_setting
from stMVC.image_processing import tiling
from stMVC.image_SSL import train_simCLR_sImage, extract_representation_simCLR_model
from stMVC.graph_embedding import RNA_encoding_train , Multi_views_attention_train, training_single_view_graph, load_calAdj_feature_data, load_calLocation_feature_data
from stMVC.image_resnet50 import Extract_representation

def train_with_argas( args ):

	basePath            = './stMVC_test_data/DLPFC_151673/'
	args.use_cuda       = args.use_cuda and torch.cuda.is_available()
	
	args.inputPath      = Path( basePath )
	args.tillingPath    = Path( basePath + 'tmp/' )
	args.tillingPath.mkdir(parents=True, exist_ok=True)
	args.outPath        = Path( basePath + 'stMVC/' ) 
	args.outPath.mkdir(parents=True, exist_ok=True)

	##load spatial transcriptomics and histological data
	#adata               = st.Read10X( args.inputPath )

	#save physical location of spots into Spot_location.csv file
	#data = { 'imagerow': adata.obs['imagerow'].values.tolist(), 'imagecol': adata.obs['imagecol'].values.tolist() }
	#df   = pd.DataFrame(data, index = adata.obs_names.tolist())
	#df.to_csv( basePath + 'spatial/' + 'Spot_location.csv' )

	##tilling histologicald data and train sinCLR model
	#tiling(adata, args.tillingPath, target_size = args.sizeImage)

	#train_simCLR_sImage(args, basePath + 'stMVC/' )
	#extract_representation_simCLR_model(args, outDir = basePath + 'stMVC/',
	#									model_file   = basePath + 'stMVC/'+ '128_0.5_200_128_500_model.pth' ) 

	## extract visual features by ResNet-50 frameowork
	#Extract_representation(args, outDir = basePath +  'stMVC/')

	## extract latent features of RNA-seq data by autoencoder-based framework
	#RNA_encoding_train(args, RNA_file = basePath + "RNA_spot_expression_count_HVG.txt", outDir = basePath + "stMVC/")


	args.fusion_type = "Attention"

	imageRep_file1   = basePath + 'stMVC/resnet50_reprensentation.csv'
	RNA_file         = basePath + 'stMVC/128_8e-05-2000-1000-50-1000-2000_RNA_AE_latent.csv'
	imageRep_file    = basePath + 'stMVC/128_0.5_200_128_simCLR_reprensentation.csv'
	imageLoc_file    = basePath + './spatial/Spot_location.csv'
	class_file       = basePath + '151673_annotation_train_test_split.csv'
	args.lr_T1       = args.lr_T2 = args.lr_T3 = 0.002

	Multi_views_attention_train(args, image_rep_file = imageRep_file, 
							    RNA_file    = RNA_file, image_loc_file    = imageLoc_file,
							    class_file1 = class_file, outDir          = basePath + 'stMVC/',
								integrate_type = "Attention", select_prop = 0.3 )

	class_data   = pd.read_csv(class_file, header = 0, index_col = 0)
	cc           = class_data.values[:,2] > 0
	aa           = list(map(int, class_data.values[:,1].tolist()))
	aa[:]        = [x - 1 for x in aa]
	bb           = class_data.values[cc, 3]>0

	rna_data1, adj_orig1, adj_train1, train_E1, test_E1, test_E_F1 = load_calAdj_feature_data(imageRep_file, RNA_file, knn = args.knn, 
		                                                                                      test_prop = 0.1, training = cc)
	rna_data_all_1, adj_orig_all_1, _, _, _, _ = load_calAdj_feature_data(imageRep_file, RNA_file, knn = args.knn )
	model_1 = training_single_view_graph(args, rna_data1, adj_orig1, adj_train1, test_E1, test_E_F1, aa, cc, bb, args.lr_T1,
										 rna_data_all_1, adj_orig_all_1, "Image_feature", basePath + 'stMVC/',  
										 0.3, class_data )

	rna_data2, adj_orig2, adj_train2, train_E2, test_E2, test_E_F2 = load_calLocation_feature_data(imageLoc_file, RNA_file, knn = args.knn, 
		                                                                                           test_prop = 0.1, training = cc)
	rna_data_all_2, adj_orig_all_2, _, _, _, _ = load_calLocation_feature_data(imageLoc_file, RNA_file, knn = args.knn )
	model_2 = training_single_view_graph(args, rna_data2, adj_orig2, adj_train2, test_E2, test_E_F2, aa, cc, bb, args.lr_T1,
										 rna_data_all_2, adj_orig_all_2, "Image_location", basePath + 'stMVC/',  
										 0.3, class_data )

	rna_data3, adj_orig3, adj_train3, train_E3, test_E3, test_E_F3 = load_calAdj_feature_data(imageRep_file1, RNA_file, knn = args.knn, 
		                                                                                      test_prop = 0.1, training = cc)
	rna_data_all_3, adj_orig_all_3, _, _, _, _ = load_calAdj_feature_data(imageRep_file1, RNA_file, knn = args.knn )
	model_3 = training_single_view_graph(args, rna_data3, adj_orig3, adj_train3, test_E3, test_E_F3, aa, cc, bb, args.lr_T1,
										 rna_data_all_3, adj_orig_all_3, "Image_feature_resnet50", basePath + 'stMVC/',  
										 0.3, class_data )

if __name__ == "__main__":

	parser  =  parameter_setting()
	args    =  parser.parse_args()
	train_with_argas(args)