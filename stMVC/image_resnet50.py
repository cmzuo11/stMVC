# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:18:05 2021

@author: chunman zuo
"""

import os
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from stMVC.modules import resnet50_model
from stMVC.image_processing import CustomDataset, test_transform

def Extract_representation( args, outDir = 'results' ):

	## extract 2048 representation from resnet50
	batch_size    = args.batch_size_I

	# data prepare
	print('step1:  ')

	total_data    = CustomDataset(imgs_path = args.tillingPath, transform = test_transform )
	total_loader  = DataLoader(total_data, batch_size=batch_size, shuffle=False, 
		                       pin_memory=True, drop_last=False)

	print('step2:  ')

	model         = resnet50_model().eval().cuda()

	# training loop
	results       = {'train_loss': [], 'test_loss': []}

	if not os.path.exists(outDir):
		os.mkdir(outDir)

	total_bar   = tqdm(total_loader)
	feature_dim = []
	barcode     = []

	for image, _, image_code in total_bar:

		image   = image.cuda(non_blocking=True)
		feature = model(image)

		feature_dim.append( feature.data.cpu().numpy() )
		barcode.append( image_code )

	feature_dim = np.concatenate(feature_dim)
	barcode     = np.concatenate(barcode)

	data_frame = pd.DataFrame(data=feature_dim, index=barcode, columns =  list(range(1, 2049)) )
	data_frame.to_csv( outDir + '/resnet50_reprensentation.csv')

