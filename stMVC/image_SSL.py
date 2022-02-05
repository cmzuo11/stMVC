# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:18:05 2021

@author: chunman zuo
"""

import os
import pandas as pd
import numpy as np
import torch
import glob
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from stMVC.modules import simCLR_model
from stMVC.image_processing import CustomDataset, train_transform_64, test_transform, train_transform_32

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, args):

	net.train().to("cuda:0")

	total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

	for pos_1, pos_2, _ in train_bar:

		pos_1, pos_2  = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

		_, out_1      = net(pos_1)
		_, out_2      = net(pos_2)
		
		out           = torch.cat([out_1, out_2], dim=0) # [2*B, D]
		sim_matrix    = torch.exp(torch.mm(out, out.t().contiguous()) / args.temperature) # [2*B, 2*B]
		mask          = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size_I, 
						 device=sim_matrix.device)).bool()
		
		sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size_I, -1) # [2*B, 2*B-1]

		# compute loss
		pos_sim    = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.temperature)
		
		pos_sim    = torch.cat([pos_sim, pos_sim], dim=0) # [2*B]
		loss       = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

		train_optimizer.zero_grad()
		loss.backward()
		train_optimizer.step()

		total_num  += args.batch_size_I
		total_loss += loss.item() * args.batch_size_I
		train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(args.current_epoch_I, args.max_epoch_I, total_loss / total_num))

	return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image

def test(net, test_data_loader, args):

	net.eval().to("cuda:1")

	total_loss, total_num, test_bar = 0.0, 0, tqdm(test_data_loader)

	with torch.no_grad():

		for pos_1, pos_2, _ in test_bar:

			pos_1, pos_2 = pos_1.to("cuda:1"), pos_2.to("cuda:1")

			_, out_1   = net(pos_1)
			_, out_2   = net(pos_2)

			out        = torch.cat([out_1, out_2], dim=0) # [2*B, D]
			sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.temperature) # [2*B, 2*B]
			mask       = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size_I, 
						  device=sim_matrix.device)).bool()
			sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size_I, -1) # [2*B, 2*B-1]

			# compute loss
			pos_sim    = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.temperature)
			pos_sim    = torch.cat([pos_sim, pos_sim], dim=0) # [2*B]
			loss       = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

			total_num  += args.batch_size_I
			total_loss += loss.item() * args.batch_size_I
			test_bar.set_description('Test Epoch: [{}/{}] Loss: {:.4f}'.format(args.current_epoch_I, args.max_epoch_I, total_loss / total_num))

	return total_loss / total_num


def train_simCLR_sImage( args, outDir = "results", sub_path = None, file_code = None):

	latent_I, k            =  args.latent_I, args.k
	batch_size, epochs     =  args.batch_size_I, args.max_epoch_I
	test_prop              =  args.test_prop
	file_list              =  None

	if file_code is not None:
		temp_file_list = []

		for z in list(range(len(file_code))):
			temp_files    =  glob.glob( sub_path + file_code[z] + "/tmp/*.jpeg" )
			temp_file_list.extend( temp_files )

		file_list = temp_file_list

	else:
		file_list         = glob.glob( str(args.tillingPath) + "/*.jpeg" )

	train_idx, test_idx   = train_test_split(np.arange(len(file_list)),
											 test_size    = 0.15,
											 random_state = 200)
	# data prepare
	print('step1:  ')
	
	if args.image_size == 64:
		train_data    = CustomDataset(imgs_path = args.tillingPath, sampling_index = train_idx, 
									  sub_path  = sub_path, file_code = file_code,
									  transform = train_transform_64 )

		test_data     = CustomDataset(imgs_path = args.tillingPath, sampling_index = test_idx, 
									  sub_path  = sub_path, file_code = file_code,
									  transform = train_transform_64)

	else:
		train_data    = CustomDataset(imgs_path = args.tillingPath, sampling_index = train_idx, 
									  sub_path  = sub_path, file_code = file_code,
									  transform = train_transform_32 )

		test_data     = CustomDataset(imgs_path = args.tillingPath, sampling_index = test_idx, 
									  sub_path  = sub_path, file_code = file_code,
									  transform = train_transform_32)
	
	train_loader  = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
							   pin_memory=True, drop_last=True)

	test_loader   = DataLoader(test_data, batch_size=batch_size, shuffle=False, 
							   pin_memory=True, drop_last=True)

	# model setup and optimizer config
	print('step2:  ')

	model      = simCLR_model(latent_I).cuda()
	optimizer  = optim.Adam(model.parameters(), lr=args.lr_I, weight_decay=1e-6)

	# training loop
	results = {'train_loss': [], 'test_loss': []}
	save_name_pre = '{}_{}_{}_{}_{}'.format(latent_I, args.temperature, k, batch_size, epochs)

	if not os.path.exists( outDir ):
		os.mkdir( outDir )

	minimum_loss    = 10000
	file_model_save = outDir + '/{}_model.pth'.format(save_name_pre)

	for epoch in range(1, epochs + 1):

		print('epoch:  '+ str(epoch))

		args.current_epoch_I = epoch
		train_loss           = train(model, train_loader, optimizer, args)
		test_loss            = test(model, test_loader, args)

		results['train_loss'].append(train_loss)
		results['test_loss'].append(test_loss)

		# save statistics
		data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
		data_frame.to_csv( outDir + '/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

		if test_loss < minimum_loss:
			minimum_loss = test_loss
			torch.save(model.state_dict(), file_model_save)

	return file_model_save


def extract_representation_simCLR_model( args, outDir = 'results', model_file = None, sub_path = None, file_code = None):

	model  =  simCLR_model(args.latent_I).cuda()
	model.load_state_dict( torch.load( model_file ) )
	model.eval()

	latent_I, k              = args.latent_I, args.k
	batch_size, epochs       = args.batch_size_I, args.max_epoch_I
	test_prop                = args.test_prop

	# data prepare
	print('step1:  ')
	total_data    = CustomDataset(imgs_path = args.tillingPath,
								  sub_path  = sub_path, file_code = file_code, 
								  transform = test_transform )

	total_loader  = DataLoader(total_data, batch_size=args.batch_size_I, shuffle=False, 
							   pin_memory=True, drop_last=False)

	print('step2:  ')

	if not os.path.exists(outDir):
		os.mkdir(outDir)

	total_bar     = tqdm(total_loader)
	feature_dim   = []
	barcode       = []

	for image, _, image_code in total_bar:

		image     = image.cuda(non_blocking=True)
		feature,_ = model(image)

		feature_dim.append( feature.data.cpu().numpy() )
		barcode.append( image_code )

	feature_dim = np.concatenate(feature_dim)
	barcode     = np.concatenate(barcode)

	save_name_pre = '{}_{}_{}_{}'.format(args.latent_I, args.temperature, args.k, args.batch_size_I)
	save_fileName = outDir + '/{}_simCLR_reprensentation.csv'.format(save_name_pre)

	data_frame    = pd.DataFrame(data=feature_dim, index=barcode, columns =  list(range(1, 2049)) )
	data_frame.to_csv( save_fileName )

	return save_fileName
