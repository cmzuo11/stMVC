# -*- coding: utf-8 -*-
"""
Created on Wed Aug  16 10:17:15 2021

@author: chunman zuo
"""

import os, sys
import argparse
import time
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils

from torch import optim
from tqdm import tqdm

from modules   import gcn_vae, VAE, AE, gat, resnet50_model, simCLR_model, Cross_Views_attention_integrate
from utilities import load_calAdj_feature_data, normalize, preprocess_graph, read_dataset, load_calLocation_feature_data

def RNA_encoding_train(args, RNA_file = None, outDir = "results"):

	args.batch_size_T   = 128
	args.epoch_per_test = 10
	args.use_cuda       = args.use_cuda and torch.cuda.is_available()

	adata, train_index, test_index, _ = read_dataset( File1 = RNA_file, transpose = True, 
													  test_size_prop = 0.1 )
	adata  = normalize( adata, filter_min_counts=True, size_factors=True,
						normalize_input=False, logtrans_input=True ) 
	
	Nsample1, Nfeature1 =  np.shape( adata.X )


	train           = data_utils.TensorDataset( torch.from_numpy( adata[train_index].X ),
												torch.from_numpy( adata.raw[train_index].X ), 
												torch.from_numpy( adata.obs['size_factors'][train_index].values ) )
	train_loader    = data_utils.DataLoader( train, batch_size = args.batch_size_T, shuffle = True )

	test            = data_utils.TensorDataset( torch.from_numpy( adata[test_index].X ),
												torch.from_numpy( adata.raw[test_index].X ), 
												torch.from_numpy( adata.obs['size_factors'][test_index].values ) )
	test_loader     = data_utils.DataLoader( test, batch_size = len(test_index), shuffle = False )

	total           = data_utils.TensorDataset( torch.from_numpy( adata.X ),
												torch.from_numpy( adata.obs['size_factors'].values ) )
	total_loader    = data_utils.DataLoader( total, batch_size = args.batch_size_T, shuffle = False )


	if args.rna_model == "VAE":
		AE_structure = [Nfeature1, 1000, 128, 10,  128, 1000, Nfeature1]
		model        = VAE( [Nfeature1, 1000, 128], hidden1 = 128, Zdim = 10, layer_d = [10, 128, 1000], 
							hidden2 = 1000, args = args, droprate = 0, type = "NB"  )

	else:
		AE_structure = [Nfeature1, 1000, 50, 1000, Nfeature1]
		model        = AE( [Nfeature1, 1000, 50], layer_d = [50, 1000], 
						   hidden1 = 1000, args = args, droprate = 0, type = "NB"  )

	if args.use_cuda:
		model.cuda()

	model.fit( train_loader, test_loader )

	save_name_pre = '{}_{}-{}'.format( args.batch_size_T, args.lr_VAET , '-'.join( map(str, AE_structure )) )
	latent_z      = model.predict(total_loader, out='z' )

	if args.rna_model == "VAE":
		torch.save(model, outDir + '/{}_RNA_VAE_model.pt'.format(save_name_pre) )
		latent_z1  = pd.DataFrame( latent_z, index= adata.obs_names ).to_csv( outDir + '/{}_RNA_VAE_latent.csv'.format(save_name_pre) ) 

	else:
		torch.save(model, outDir + '/{}_RNA_AE_model.pt'.format(save_name_pre) )
		latent_z1  = pd.DataFrame( latent_z, index= adata.obs_names ).to_csv( outDir + '/{}_RNA_AE_latent.csv'.format(save_name_pre) ) 


def training_single_view_graph(args, rna_data, adj_orig, adj_train, test_E, test_E_F, aa, cc, bb, learning_rate,
							   rna_data_all, adj_orig_all, pattern = "Image_feature", outDir = "results",  
							   select_prop = 0.1, class_data1 = None ):

	_, feat_dim  = rna_data.values.shape
	
	if args.graph_model == "GAT":
		model_s    = gat(args, feat_dim, args.latent_T1, args.latent_T2, 
						 dropout = 0, nheads= args.attention_head).cuda()
		for_save = "GAT" + "_" + str(args.attention_head)

	else:
		model_s    = gcn_vae(args, feat_dim, args.latent_T1, args.latent_T2, 
							 dropout = 0).cuda()
		for_save = "GCN"

	rna_data_exp = torch.FloatTensor(rna_data.values[cc,:]).cuda()

	model_s.fit( rna_data_exp, adj_orig, adj_train, test_E, test_E_F, None, 
				 None, learning_rate, np.array( aa )[cc], training_info = bb )

	rna_data_exp_a             = torch.FloatTensor(rna_data_all.values).cuda()
	adj_org_norm_a             = preprocess_graph(adj_orig_all).cuda()
	_, mu, _,class_prediction  = model_s( rna_data_exp_a, adj_org_norm_a )

	model_s.evaluation_metrics( mu, class_data1.values[:,1]>0, np.array( aa )  )
	model_s.evaluation_classification( np.array( aa ), class_prediction, class_data1.values[:,1]>0 )

	save_name_pre = '{}_{}_{}_{}_{}_{}'.format( args.latent_T1, args.latent_T2, args.max_epoch_T, learning_rate, args.lr_VAET, args.knn)
	data_frame    = pd.DataFrame(data=mu.data.cpu().numpy(), index=rna_data_all.index, 
								 columns =  list(range(1, (args.latent_T2+1))) )

	data_frame1   = pd.DataFrame( data=class_prediction.data.cpu().numpy(), index=rna_data_all.index )

	if pattern == "Image_feature":
		data_frame.to_csv( outDir + '/{}_{}_image_feature_rep_0.7_single_{}.csv'.format(save_name_pre, for_save, str(select_prop) ) )
		data_frame1.to_csv( outDir + '/{}_{}_image_feature_class_prediction_0.7_single_{}.csv'.format(save_name_pre, for_save, str(select_prop) ) )
		torch.save(model_s.state_dict(), outDir + '/{}_{}_image_feature_model_0.7_single_{}.pth'.format(save_name_pre, for_save, str(select_prop) ))

	elif pattern == "Image_location":
		data_frame.to_csv( outDir + '/{}_{}_image_location_rep_0.7_single_{}.csv'.format(save_name_pre, for_save, str(select_prop) ) )
		data_frame1.to_csv( outDir + '/{}_{}_image_location_class_prediction_0.7_single_{}.csv'.format(save_name_pre, for_save, str(select_prop) ) )
		torch.save(model_s.state_dict(), outDir + '/{}_{}_image_location_model_0.7_single_{}.pth'.format(save_name_pre, for_save, str(select_prop) ))

	elif pattern == "RNA_similarity": # adjancy matrix by rna similarity
		data_frame.to_csv( outDir + '/{}_{}_rna_similarity_rep_0.7_single_{}.csv'.format(save_name_pre, for_save, str(select_prop) ) )
		data_frame1.to_csv( outDir + '/{}_{}_rna_similarity_class_prediction_0.7_single_{}.csv'.format(save_name_pre, for_save, str(select_prop) ) )
		torch.save(model_s.state_dict(), outDir + '/{}_{}_rna_similarity_model_0.7_single_{}.pth'.format(save_name_pre, for_save, str(select_prop) ))

	elif pattern == "Image_feature_resnet50":
		data_frame.to_csv( outDir + '/{}_{}_image_feature_resnet50_rep_0.7_single_{}.csv'.format(save_name_pre, for_save, str(select_prop) ) )
		data_frame1.to_csv( outDir + '/{}_{}_image_feature_resnet50_class_prediction_0.7_single_{}.csv'.format(save_name_pre, for_save, str(select_prop) ) )
		torch.save(model_s.state_dict(), outDir + '/{}_{}_image_feature_resnet50_model_0.7_single_{}.pth'.format(save_name_pre, for_save, str(select_prop) ))

	else:
		data_frame.to_csv( outDir + '/{}_{}_image_rna_similarity_rep_0.7_single_{}.csv'.format(save_name_pre, for_save, str(select_prop) ) )
		data_frame1.to_csv( outDir + '/{}_{}_image_rna_similarity_class_prediction_0.7_single_{}.csv'.format(save_name_pre, for_save, str(select_prop) ) )
		torch.save(model_s.state_dict(), outDir + '/{}_{}_image_rna_similarity_model_0.7_single_{}.pth'.format(save_name_pre, for_save, str(select_prop) ))

	return model_s


def Multi_views_attention_train(args, image_rep_file = None, RNA_file = None, image_loc_file = None, 
								class_file1 = None, outDir = "results", integrate_type = "Attention",
								select_prop = 0.7 ):

	class_data1  = pd.read_csv(class_file1, header = 0, index_col = 0)
	cc           = class_data1.values[:,2]>0

	rna_data1, adj_orig1, adj_train1, train_E1, test_E1, test_E_F1 = load_calAdj_feature_data(image_rep_file, RNA_file, knn = args.knn, 
																							  test_prop = 0.1, training = cc)
	rna_data2, adj_orig2, adj_train2, train_E2, test_E2, test_E_F2 = load_calLocation_feature_data(image_loc_file, RNA_file, knn = args.knn, 
																								   test_prop = 0.1, training = cc)

	aa           = list(map(int, class_data1.values[:,1].tolist()))
	aa[:]        = [x - 1 for x in aa]
	bb           = class_data1.values[cc, 3]>0

	rna_data_all_1, adj_orig_all_1, _, _, _, _ = load_calAdj_feature_data(image_rep_file, RNA_file, knn = args.knn )
	rna_data_all_2, adj_orig_all_2, _, _, _, _ = load_calLocation_feature_data(image_loc_file, RNA_file, knn = args.knn )

	model  = Cross_Views_attention_integrate(args, rna_data1.values.shape[1], args.latent_T1, args.latent_T2, 
											 rna_data2.values.shape[1], args.latent_T1, args.latent_T2, 
											 nClass = args.cluster_pre, nheads1 =args.attention_head, 
											 nheads2 = args.attention_head, model_pattern = "GAT",
											 max_iteration  = args.max_iter, class_label1 = np.array(aa)[cc],
											 class_label2   = np.array(aa)[cc], training  = bb,
											 integrate_type = integrate_type )

	model.fit_model(rna_data1.values[cc,:], adj_orig1, adj_train1, test_E1, test_E_F1,
		            rna_data2.values[cc,:], adj_orig2, adj_train2, test_E2, test_E_F2,
		            rna_data_all_1.values, adj_orig_all_1, rna_data_all_2.values, adj_orig_all_2,
		            np.array(aa), None)

	save_name_pre = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format( args.latent_T1, args.latent_T2, args.max_epoch_T, 
														 args.lr_T1, args.lr_T2, args.lr_crossView, args.beta_pa,
														 args.knn, integrate_type )

	torch.save( model.state_dict(), outDir + '/{}_{}_2-view_model_split_{}_temp_new.pth'.format(save_name_pre, "GAT", str(select_prop) ) )
	
	lamda, mu_robust1, class_prediction = model( rna_data_all_1.values, adj_orig_all_1, 
												 rna_data_all_2.values, adj_orig_all_2 )

	data_frame  = pd.DataFrame(data=mu_robust1.data.cpu().numpy(), index=rna_data1.index ).to_csv( outDir + 
							   '/{}_{}_2-view_robust_representation_split_{}_temp_new.csv'.format(save_name_pre, "GAT", str(select_prop) ) ) 

	if integrate_type != "NN":
		data_frame  = pd.DataFrame(data=lamda.data.cpu().numpy(), index=rna_data1.index ).to_csv( outDir + 
								   '/{}_{}_2-view_lamda_coefficients_split_{}_temp_new.csv'.format(save_name_pre, "GAT", str(select_prop) ) ) 

	data_frame  = pd.DataFrame(data=class_prediction.data.cpu().numpy(), index=rna_data1.index ).to_csv( outDir + 
							   '/{}_{}_2-view_class_prediction_split_{}_temp_new.csv'.format(save_name_pre, "GAT", str(select_prop) ) ) 

