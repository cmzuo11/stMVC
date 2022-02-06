# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:16:25 2021

@author: chunman zuo
"""

import numpy as np
import pandas as pd
import os
import time
import torch
import random
from pathlib import Path

from stMVC.utilities import parameter_setting
from stMVC.graph_embedding import Multi_views_attention_train

def train_with_argas( args ):

	start            = time.time()

	args.fusion_type = "Attention"
	args.outPath     = Path(args.basePath + 'stMVC/')
	args.outPath.mkdir(parents=True, exist_ok=True)

	RNA_file         = args.basePath + 'stMVC/128_8e-05-2000-1000-50-1000-2000_RNA_AE_latent.csv'
	imageRep_file    = args.basePath + 'stMVC/128_0.5_200_128_simCLR_reprensentation.csv'
	imageLoc_file    = args.basePath + 'spatial/Spot_location.csv'

	class_file       = args.basePath + 'Annotation_train_test_split.csv'
	args.lr_T1       = args.lr_T2 = args.lr_T3 = 0.002

	args.use_cuda    = args.use_cuda and torch.cuda.is_available()

	Multi_views_attention_train(args, imageRep_file, RNA_file, 
		                        imageLoc_file, class_file, 
		                        args.basePath + 'stMVC/', args.fusion_type )

	duration = time.time() - start
	print('Finish training, total time is: ' + str(duration) + 's' )

if __name__ == "__main__":

	parser  =  parameter_setting()
	args    =  parser.parse_args()
	train_with_argas(args)
