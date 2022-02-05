# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:16:25 2021

@author: chunman zuo
"""
import stlearn as st

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

	args.fusion_type = "Attention"

	#imageRep_file1   = args.basePath + 'stMVC/resnet50_reprensentation.csv'
	RNA_file         = args.basePath + 'stMVC/128_8e-05-2000-1000-50-1000-2000_RNA_AE_latent.csv'
	imageRep_file    = args.basePath + 'stMVC/128_0.5_200_128_simCLR_reprensentation.csv'
	imageLoc_file    = args.basePath + './spatial/Spot_location.csv'
	class_file       = args.basePath + '151673_annotation_train_test_split.csv'
	args.lr_T1       = args.lr_T2 = args.lr_T3 = 0.002

	Multi_views_attention_train(args, image_rep_file = imageRep_file, 
							    RNA_file    = RNA_file, image_loc_file    = imageLoc_file,
							    class_file1 = class_file, outDir          = args.basePath + 'stMVC/',
								integrate_type = "Attention" )

if __name__ == "__main__":

	parser  =  parameter_setting()
	args    =  parser.parse_args()
	train_with_argas(args)