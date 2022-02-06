# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:13:00 2021

@author: chunman zuo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import torch
import random

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
#import stlearn as st

from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, pairwise_distances
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, scale
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine


def parameter_setting():
	
	parser      = argparse.ArgumentParser(description='Spatial transcriptomics analysis')
	BasePath    = './stMVC_test_data/DLPFC_151673/'

	parser.add_argument('--basePath', '-bp', type=str, default = BasePath, help='base path for the output of 10X pipeline')
	
	parser.add_argument('--inputPath',   '-IP', type = str, default = None,    help='data directory')
	parser.add_argument('--tillingPath', '-TP', type = str, default = None,  help='image data directory')
	parser.add_argument('--outPath', '-od', type=str, default = None, help='Output path')

	parser.add_argument('--jsonFile', '-jFile', type=str, default = None, help='image cell segmentation file')
	
	parser.add_argument('--weight_decay', type=float, default = 1e-6, help='weight decay')
	parser.add_argument('--eps', type=float, default = 0.01, help='eps')

	parser.add_argument('--cluster_pre', '-clup', type=int, default=7, help='predefined cluster for scRNA')
	parser.add_argument('--geneClu', '-gClu', type=list, default = None, help='predefined gene cluster for scRNA')
	parser.add_argument('--beta_pa', '-bePa', type=float, default = 0.0005, help='parameter for robust representation loss')
	
	parser.add_argument('--batch_size_T', '-bT', type=int, default=128, help='Batch size for transcriptomics data')

	parser.add_argument('--batch_size_I', '-bI', type=int, default=128, help='Batch size for spot image data')
	parser.add_argument('--image_size', '-iS', type=int, default=32, help='image size for spot image data')

	parser.add_argument('--latent_T1', '-lT1',type=int, default=25, help='Feature dim1 for latent vector for transcriptomics data')
	parser.add_argument('--latent_T2', '-lT2',type=int, default=10, help='Feature dim2 for latent vector for transcriptomics data')
	parser.add_argument('--latent_I', '-lI',type=int, default=128, help='Feature dim for latent vector for spot image data')

	parser.add_argument('--max_epoch_T', '-meT', type=int, default=500, help='Max epoches for transcriptomics data')
	parser.add_argument('--max_epoch_I', '-meI', type=int, default=500, help='Max epoches for spot image data')
	parser.add_argument('--current_epoch_I', '-curEI', type=int, default=0, help='current epoches for spot image data')

	parser.add_argument('--lr_T1', type=float, default = 0.002, help='Learning rate of SGATE model for graph constructed by vision features')
	parser.add_argument('--lr_T2', type=float, default = 0.002, help='Learning rate of SGATE model for graph constructed by spatial location data')
	parser.add_argument('--lr_T3', type=float, default = 0.002, help='Learning rate for multi-view graph collaborative learning model')
	parser.add_argument('--lr_AET', type=float, default = 8e-05, help='Learning rate of AE model for transcriptomics data')
	parser.add_argument('--lr_AET_F', type=float, default = 0.00001, help='final learning rate of AE model for transcriptomics data')
	parser.add_argument('--lr_I', type=float, default = 0.0001, help='Learning rate for spot image data')

	parser.add_argument('--rna_model', '-rnaModel', type=str, default = 'AE', help='extract RNA information')
	parser.add_argument('--image_model', '-imageModel', type=str, default = 'AE', help='extract image information')
	parser.add_argument('--graph_model', '-graphModel', type=str, default = 'GAT', help='graph attention model (GAT or GCN)')
	parser.add_argument('--attention_head', '-attentionHead', type=int, default = 2, help='the number of attention heads (GAT or GCN)')
	parser.add_argument('--fusion_type', '-fusionType', type=str, default = "Attention", help='the type of multi-view graph fusion')

	parser.add_argument('--use_cuda', dest='use_cuda', default=True, action='store_true', help=" whether use cuda(default: True)")
	
	parser.add_argument('--seed', type=int, default=200, help='Random seed for repeat results')

	parser.add_argument('--max_iteration', '-mi', type=int, default=3000, help='Max iteration')
	parser.add_argument('--anneal_epoch', '-ae', type=int, default=200, help='Anneal epoch')
	parser.add_argument('--epoch_per_test', '-ept', type=int, default=5, help='Epoch per test')
	parser.add_argument('--max_ARI', '-ma', type=int, default=-200, help='initial ARI')

	parser.add_argument('--max_iter', '-maIter', type=int, default=10, help='max iteration')
	parser.add_argument('--knn', '-KNN', type=int, default=7, help='K nearst neighbour')

	parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
	parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
	parser.add_argument('--sizeImage', type=int, default=32, help='Random seed for repeat results')

	parser.add_argument('--test_prop', default=0.05, type=float, help='the proportion data for testing')
	
	return parser


def normalize( adata, filter_min_counts=True, size_factors=True, 
			   normalize_input=False, logtrans_input=True):

	if filter_min_counts:
		sc.pp.filter_genes(adata, min_counts=1)
		sc.pp.filter_cells(adata, min_counts=1)

	if size_factors or normalize_input or logtrans_input:
		adata.raw = adata.copy()
	else:
		adata.raw = adata

	if logtrans_input:
		sc.pp.log1p(adata)

	if size_factors:
		#adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
		adata.obs['size_factors'] = np.log( np.sum( adata.X, axis = 1 ) )
	else:
		adata.obs['size_factors'] = 1.0

	if normalize_input:
		sc.pp.scale(adata)

	return adata


def load_HSG_SLG_by_feature_data( image_loc_file  = None, 
								  rna_file        = None,
								  class_file      = None, 
								  knn             = 7, 
								  disatnce_method = "cosine", 
								  test_prop       = 0.1, 
								  training        = None ):

	## visual features or spatial location file
	image_loc_in = pd.read_csv(image_loc_file, header = 0, index_col = 0)

	if rna_file.find('csv') != -1:
		rna_exp  = pd.read_csv(rna_file, header = 0, index_col = 0)
	else:
		rna_exp  = pd.read_table(rna_file, header = 0, index_col = 0)

	class_data   = pd.read_csv(class_file, header = 0, index_col = 0)

	image_loc_in = image_loc_in.reindex(class_data.index)
	rna_exp      = rna_exp.reindex(class_data.index)

	if training is not None:
		dist_out = pairwise_distances(image_loc_in.values[training,:], metric = disatnce_method)

	else:
		dist_out = pairwise_distances(image_loc_in.values, metric = disatnce_method)

	row_index    = []
	col_index    = []

	sorted_knn   = dist_out.argsort(axis=1)

	for index in list(range( np.shape(dist_out)[0] )):
		col_index.extend( sorted_knn[index, :knn].tolist() )
		row_index.extend( [index] * knn )

	adj        = sp.coo_matrix( (np.ones( len(row_index) ), (row_index, col_index) ), 
								shape=( np.shape(dist_out)[0], np.shape(dist_out)[0] ), dtype=np.float32 )

	adj_orig   = adj
	adj_orig   = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)

	adj_orig.eliminate_zeros()

	adj_train, train_edges, test_edges, test_edges_false = mask_test_edges(adj, test_pro = test_prop)

	return rna_exp, class_data, adj_orig, adj_train, train_edges, test_edges, test_edges_false


def mask_test_edges(adj, test_pro = 0.1, val_pro = 0.2):
	# split edge into training and testing set

	# Remove diagonal elements
	adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
	adj.eliminate_zeros()
	# Check that diag is zero:
	assert np.diag(adj.todense()).sum() == 0

	adj_triu  = sp.triu(adj)
	adj_tuple = sparse_to_tuple(adj_triu)
	edges     = adj_tuple[0]
	edges_all = sparse_to_tuple(adj)[0]

	num_test     = int(np.floor(edges.shape[0] * test_pro ))
	num_val      = int(np.floor(edges.shape[0] * val_pro ))

	all_edge_idx = np.arange(edges.shape[0])

	np.random.shuffle(all_edge_idx)
	val_edge_idx  = all_edge_idx[:num_val]
	test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
	test_edges    = edges[test_edge_idx]
	val_edges     = edges[val_edge_idx]
	train_edges   = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

	def ismember(a, b, tol=5):
		rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
		return np.any(rows_close)

	test_edges_false = []
	while len(test_edges_false) < len(test_edges):
		idx_i = np.random.randint(0, adj.shape[0])
		idx_j = np.random.randint(0, adj.shape[0])
		if idx_i == idx_j:
			continue
		if ismember([idx_i, idx_j], edges_all):
			continue
		if test_edges_false:
			if ismember([idx_j, idx_i], np.array(test_edges_false)):
				continue
			if ismember([idx_i, idx_j], np.array(test_edges_false)):
				continue
		test_edges_false.append([idx_i, idx_j])

	val_edges_false = []
	while len(val_edges_false) < len(val_edges):
		idx_i = np.random.randint(0, adj.shape[0])
		idx_j = np.random.randint(0, adj.shape[0])
		if idx_i == idx_j:
			continue
		if ismember([idx_i, idx_j], train_edges):
			continue
		if ismember([idx_j, idx_i], train_edges):
			continue
		if ismember([idx_i, idx_j], val_edges):
			continue
		if ismember([idx_j, idx_i], val_edges):
			continue
		if val_edges_false:
			if ismember([idx_j, idx_i], np.array(val_edges_false)):
				continue
			if ismember([idx_i, idx_j], np.array(val_edges_false)):
				continue
		if ~ismember([idx_i,idx_j],edges_all) and ~ismember([idx_j,idx_i],edges_all):
			val_edges_false.append([idx_i, idx_j])
		else:
			# Debug
			print(str(idx_i)+" "+str(idx_j))

	data      = np.ones(train_edges.shape[0])

	# Re-build adj matrix
	adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
	adj_train = adj_train + adj_train.T

	# NOTE: these edge lists only contain single direction of edge!
	return adj_train, train_edges, test_edges, test_edges_false


def evaluation_clustering_metrics(args, repren_file = None, class_file = None):

	rep_data   = pd.read_csv(repren_file, header = 0, index_col = 0) 
	clas_data  = pd.read_table(class_file, header = 0, index_col = 0)

	training   = clas_data.values[:,1]>0
	aa         = list(map(int, clas_data.values[:,1].tolist()))
	aa[:]      = [x - 1 for x in aa]

	kmeans     = KMeans( n_clusters = args.cluster_pre, n_init = 5, random_state = 200 )
	ARI_score  = -100
	NMI_score  = -100

	pred_z1    = kmeans.fit_predict( rep_data.values )
	NMI_score  = round( normalized_mutual_info_score( np.array(aa)[training].tolist(), 
													  pred_z1[training],  average_method='max' ), 3 )

	ARI_score  = round( metrics.adjusted_rand_score( np.array(aa)[training].tolist(), pred_z1[training] ), 3 )

	print('clustering ARI score: ' + str(ARI_score) + ' NMI score: ' + str(NMI_score) )
	return ARI_score, NMI_score


def sparse_to_tuple(sparse_mx):
	# convert graph to coordiated and values
	if not sp.isspmatrix_coo(sparse_mx):
		sparse_mx = sparse_mx.tocoo()
	coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
	values = sparse_mx.data
	shape  = sparse_mx.shape
	return coords, values, shape


def normalize_adj(adj):
	"""Row-normalize sparse matrix"""
	adj    = sp.coo_matrix(adj)
	adj_   = adj + sp.eye(adj.shape[0])
	rowsum = np.array(adj_.sum(1))

	degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
	adj_normalized      = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
	# return sparse_to_tuple(adj_normalized)
	return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	# Predict on test set of edges
	adj_rec = np.dot(emb, emb.T)
	preds   = []
	pos     = []
	for e in edges_pos:
		preds.append(sigmoid(adj_rec[e[0], e[1]]))
		pos.append(adj_orig[e[0], e[1]])

	preds_neg = []
	neg       = []
	for e in edges_neg:
		preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
		neg.append(adj_orig[e[0], e[1]])

	preds_all  = np.hstack([preds, preds_neg])
	labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
	roc_score  = roc_auc_score(labels_all, preds_all)
	ap_score   = average_precision_score(labels_all, preds_all)

	return roc_score, ap_score


def preprocess_graph(adj):
	adj    = sp.coo_matrix(adj)
	adj_   = adj + sp.eye(adj.shape[0])
	rowsum = np.array(adj_.sum(1))
	degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
	adj_normalized      = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
	# return sparse_to_tuple(adj_normalized)
	return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
	"""Convert a scipy sparse matrix to a torch sparse tensor."""
	# sparse_mx = sparse_mx.tocoo().astype(np.float64)
	sparse_mx = sparse_mx.tocoo().astype(np.float32)
	indices   = torch.from_numpy( np.vstack((sparse_mx.row, 
								  sparse_mx.col)).astype(np.int64))
	values    = torch.from_numpy(sparse_mx.data)
	shape     = torch.Size(sparse_mx.shape)
	# return torch.sparse.DoubleTensor(indices, values, shape)
	return torch.sparse.FloatTensor(indices, values, shape)


def adjust_learning_rate(init_lr, optimizer, iteration, max_lr, adjust_epoch):

	lr = max(init_lr * (0.9 ** (iteration//adjust_epoch)), max_lr)
	for param_group in optimizer.param_groups:
		param_group["lr"] = lr

	return lr   


def save_checkpoint(model, folder='./saved_model/', filename='model_best.pth.tar'):
	if not os.path.isdir(folder):
		os.mkdir(folder)

	torch.save(model.state_dict(), os.path.join(folder, filename))

def load_checkpoint(file_path, model, use_cuda=False):

	if use_cuda:
		device = torch.device( "cuda" )
		model.load_state_dict( torch.load(file_path) )
		model.to(device)
		
	else:
		device = torch.device('cpu')
		model.load_state_dict( torch.load(file_path, map_location=device) )

	model.eval()
	return model


def read_dataset( File1 = None, File2 = None,  transpose = True, test_size_prop = 0.15, state = 0 ):

	### File1 for raw reads count 
	if File1 is not None:
		adata = sc.read(File1)

		if transpose:
			adata = adata.transpose()
	else:
		adata = None
	
	### File2 for cell group information
	label_ground_truth = []

	if state == 0 :

		if File2 is not None:

			Data2 = pd.read_csv( File2, header=0, index_col=0 )
			## preprocessing for latter evaluation

			group = Data2['Group'].values

			for g in group:
				g = int(g.split('Group')[1])
				label_ground_truth.append(g)

		else:
			label_ground_truth =  np.ones( len( adata.obs_names ) )

	if test_size_prop > 0 :
		train_idx, test_idx = train_test_split(np.arange(adata.n_obs), 
											   test_size = test_size_prop, 
											   random_state = 200)
		spl = pd.Series(['train'] * adata.n_obs)
		spl.iloc[test_idx]  = 'test'
		adata.obs['split']  = spl.values
		
	else:
		train_idx, test_idx = list(range( adata.n_obs )), list(range( adata.n_obs ))

		spl = pd.Series(['train'] * adata.n_obs)
		adata.obs['split']       = spl.values
		
	adata.obs['split'] = adata.obs['split'].astype('category')
	adata.obs['Group'] = label_ground_truth
	adata.obs['Group'] = adata.obs['Group'].astype('category')
	
	print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))
	
	### here, adata with cells * features
	return adata, train_idx, test_idx, label_ground_truth

def data_normalization( File1, outFile, reverse = False ):

	scaler          = MinMaxScaler(feature_range=(0, 1))
	image_latent    = pd.read_csv(File1, header=0, index_col=0)

	if reverse:
		image_norm  = scaler.fit_transform( image_latent.values.T )
		latent_z1   = pd.DataFrame( image_norm.T, index= image_latent.index, 
									columns = image_latent.columns ).to_csv( outFile ) 

	else:
		image_norm  = scaler.fit_transform( image_latent.values )
		latent_z1   = pd.DataFrame( image_norm, index= image_latent.index, 
									columns = image_latent.columns ).to_csv( outFile ) 
