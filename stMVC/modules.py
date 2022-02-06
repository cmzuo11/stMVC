# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:17:15 2021

@author: chunman zuo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import collections
import os
import time
import math
import random
import torch.utils.data as data_utils
import scipy.sparse as sp

from sklearn.cluster import KMeans
from torch import optim
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torchvision import models
from collections import OrderedDict
from torch.distributions import Normal, kl_divergence as kl
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score

from stMVC.layers import GraphConvolution, build_multi_layers, Encoder, Decoder_logNorm_NB, Decoder, GraphAttentionLayer
from stMVC.layers import CrossViewAttentionLayer, Integrate_multiple_view_model
from stMVC.loss_function import log_nb_positive, mse_loss, loss_function, supervised_multiple_loss_function, loss_function_total, loss_function_total_2
from stMVC.utilities import adjust_learning_rate, preprocess_graph, get_roc_score
from stMVC.loss_function import log_nb_positive
		
class gat(nn.Module):
	def __init__(self, args, nfeat, nhid, nclass, dropout = 0.0, alpha = 0.2, nheads =2):
		"""Dense version of GAT."""
		super(gat, self).__init__()
		self.dropout = dropout

		self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
		for i, attention in enumerate(self.attentions):
			self.add_module('attention_{}'.format(i), attention)

		self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
		self.dc      = InnerProductDecoder(dropout, act=lambda x: x)

		self.W1      = nn.Parameter(torch.empty(size=(nclass, args.cluster_pre)))
		nn.init.xavier_uniform_(self.W1.data, gain=1.414)

		self.args    = args

	def inference(self, x, adj):
		x = F.dropout(x, self.dropout, training=self.training)
		x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
		x = F.dropout(x, self.dropout, training=self.training)
		x = F.elu(self.out_att(x, adj))

		return self.dc(x), x, None, F.softmax(torch.mm(x, self.W1), dim = 1)

	def graph_processing(self, adj_train):

		adj        = adj_train

		# Some preprocessing
		adj_norm   = preprocess_graph(adj).cuda()
		adj_label  = adj_train + sp.eye(adj_train.shape[0])
		adj_label  = torch.FloatTensor(adj_label.toarray()).cuda()

		pos_weight = torch.as_tensor( ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()) ) 
		norm       = torch.as_tensor( adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2) )

		return adj_norm, adj_label, pos_weight, norm


	def forward(self, x, adj):
		reconstruction, mu, logvar, class_pre = self.inference(x, adj)
		
		return reconstruction, mu, logvar, class_pre


	def fit(self, rna_data, adj_orig, adj_train, test_E, test_E_F, 
			lamda = None, robust_rep = None, lr_model = 0.002,
			true_class = None, training_info = None, loc = 0, penalty = 1  ):

		n_spot, _    = rna_data.size()
		adj_norm, adj_label, pos_weight, norm = self.graph_processing( adj_train )
		#optimizer    = optim.Adam( self.parameters(), lr=lr_model )

		params    = filter(lambda p: p.requires_grad, self.parameters())
		optimizer = optim.Adam( params, lr = lr_model, weight_decay = self.args.weight_decay, eps = self.args.eps )

		_, mu, _, _  = self.inference(rna_data, adj_norm)
		self.evaluation_metrics(mu, training_info, true_class)

		hidden_emb   = None
		train_bar    = tqdm(range(self.args.max_epoch_T))
		epoch        = 0
		train_loss   = []

		self.train()

		for index in train_bar:
			epoch = epoch + 1

			t     = time.time()

			optimizer.zero_grad()
			recovered, mu, logvar, class_pre  = self.inference(rna_data, adj_norm)
			
			cost, KLD, robust_loss, pre_cost  = loss_function(preds      = recovered, labels=adj_label,
															  mu         = mu,        logvar=logvar, 
															  n_nodes    = torch.as_tensor(n_spot).cuda(),
															  norm       = norm,  pos_weight=pos_weight,
															  lamda      = lamda, robust_rep=robust_rep,
															  prediction = class_pre, true_class = true_class,
															  training   = training_info, loc = loc)

			loss = cost + KLD + penalty * self.args.beta_pa * robust_loss +  8*pre_cost
			
			loss.backward()
			optimizer.step()

			hidden_emb        = mu.data.cpu().numpy()
			roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, test_E, test_E_F)

			train_bar.set_description('Train Epoch: [{}/{}] graph_loss: {:.3f} robust_rep: {:.3f} pre_cost: {:.4f}'.format(epoch, 
									  self.args.max_epoch_T, cost.item(), penalty * self.args.beta_pa * robust_loss.item(),  8*pre_cost.item() ))

			train_loss.append( loss.item() )

		self.eval()

		print("Optimization Finished!")

		_, mu, _, _  = self.inference(rna_data, adj_norm)
		self.evaluation_metrics(mu, training_info, true_class)

		return sum(train_loss)/len(train_loss)

	def evaluation_metrics(self, repren = None, training = None, true_class = None):

		kmeans     = KMeans( n_clusters = self.args.cluster_pre, n_init = 5, random_state = 200 )
		ARI_score, NMI_score = -100, -100

		if training is not None:
			pred_z1    = kmeans.fit_predict( repren[training,:].data.cpu().numpy() )
			NMI_score  = round( normalized_mutual_info_score( true_class[training].tolist(), 
														      pred_z1,  average_method='max' ), 3 )
			ARI_score  = round( metrics.adjusted_rand_score( true_class[training].tolist(), pred_z1 ), 3 )

		else:
			pred_z1    = kmeans.fit_predict( repren.data.cpu().numpy() )
			NMI_score  = round( normalized_mutual_info_score( true_class.tolist(), 
														      pred_z1,  average_method='max' ), 3 )
			ARI_score  = round( metrics.adjusted_rand_score( true_class.tolist(), pred_z1 ), 3 )

		print('single clustering ARI score: ' + str(ARI_score) + ' NMI score: ' + str(NMI_score) )
		return ARI_score, NMI_score

	def evaluation_classification(self, true, predict, training = None):

		ARI_score, NMI_score = -100, -100

		prediction    = predict.data.cpu().numpy().tolist()
		predict_class = np.array(list(map(lambda x: x.index(max(x)), prediction)))

		if training is not None:
			ARI_score = round(metrics.adjusted_rand_score( true[training].tolist(), predict_class[training].tolist() ),3)
			NMI_score = round(normalized_mutual_info_score( true[training].tolist(), predict_class[training].tolist() ),3)

		else:
			ARI_score = round(metrics.adjusted_rand_score( true.tolist(), predict_class.tolist() ),3)
			NMI_score = round(normalized_mutual_info_score( true.tolist(), predict_class.tolist() ),3)

		print( 'single classification ARI score: ' + str(ARI_score) + ' NMI score: ' + str(NMI_score) )
		return ARI_score, NMI_score


class InnerProductDecoder(Module):
	"""Decoder for using inner product for prediction."""

	def __init__(self, dropout, act=torch.sigmoid):
		super(InnerProductDecoder, self).__init__()
		self.dropout = dropout
		self.act = act

	def forward(self, z):
		z = F.dropout(z, self.dropout, training=self.training)
		adj = self.act(torch.mm(z, z.t()))
		return adj

class Cross_Views_attention_integrate(Module):

	def __init__(self, args, nfeat1, nhid1, nclass1, nfeat2, nhid2, nclass2, 
				 nClass, nheads1 =1, nheads2 =1, dropout  = 0.1, alpha  = 0.2, 
				 model_pattern = "GAT", max_iteration = 5,  class_label = None, 
				 training = None, integrate_type = "Attention"):

		super(Cross_Views_attention_integrate, self).__init__()

		self.view1  = gat(args, nfeat1, nhid1, nclass1, dropout = dropout, alpha = alpha, nheads = nheads1).cuda()
		self.view2  = gat(args, nfeat2, nhid2, nclass2, dropout = 0, alpha = alpha, nheads = nheads1).cuda()

		if integrate_type == "Attention":
			self.crossview  = CrossViewAttentionLayer(nclass1, nClass, dropout = 0.0, alpha = 0.2).cuda()

		else: #integrate_type == "Mean"
			self.crossview  = Integrate_multiple_view_model(nclass1, nClass, dropout = 0.0, alpha = 0.2, type = "Prop").cuda()

		self.max_iteration = max_iteration
		self.class_label   = class_label
		self.training_int  = training
		self.args          = args

	def fit_model_collaborative(self, rna_data1 = None, adj_train1 = None, rna_data2  = None, adj_train2  = None, 
								lamda1    = None,   mu_robust1= None, lr_model        = 0.002, 
								training  = None, true_class  = None,  type           = "context"):

		n_spot, _      = np.shape( rna_data1 )
		rna_data_exp1  = torch.FloatTensor(rna_data1).cuda()
		rna_data_exp2  = torch.FloatTensor(rna_data2).cuda()

		adj_norm1, adj_label1, pos_weight1, norm1 = self.view1.graph_processing( adj_train1 )
		adj_norm2, adj_label2, pos_weight2, norm2 = self.view2.graph_processing( adj_train2 )

		params    = filter(lambda p: p.requires_grad, self.parameters())
		optimizer = optim.Adam( params, lr = lr_model, weight_decay = self.args.weight_decay, eps = self.args.eps )

		self.train()

		train_bar   = tqdm(range(self.args.max_epoch_T))
		epoch       = 0
		loss_record = 0
		train_loss  = 0

		for index in train_bar:

			epoch = epoch + 1
			t     = time.time()

			optimizer.zero_grad()

			recovered1, mu1, logvar1, class_pre1  = self.view1.inference(rna_data_exp1, adj_norm1)
			recovered2, mu2, logvar2, class_pre2  = self.view2.inference(rna_data_exp2, adj_norm2)

			cost, robust_loss, KLD, pre_cost   = loss_function_total_2(recovered1, adj_label1, mu1, logvar1, class_pre1, norm1, pos_weight1, 
																	   recovered2, adj_label2, mu2, logvar2, class_pre2, norm2, pos_weight2,
																	   n_nodes    = torch.as_tensor(n_spot).cuda(), lamda = lamda1,
																	   robust_rep = mu_robust1, true_class = true_class, training = training)
			if type == "context":
				lamda, mu_robust, class_prediction = self.crossview( mu1, mu2 )
				classify_loss  = supervised_multiple_loss_function( class_prediction,  true_class, training)

			else: # view-specific
				classify_loss  = torch.tensor(0.0)

			loss  = cost + KLD + self.args.beta_pa * robust_loss + 10*pre_cost + 100*classify_loss

			loss.backward()
			optimizer.step()

			train_bar.set_description('Train Epoch: [{}/{}] graph_loss: {:.3f} robust_rep: {:.3f} pre_cost: {:.4f} classi_loss: {:.3f}'.format(epoch, 
									  self.args.max_epoch_T, cost.item(), self.args.beta_pa * robust_loss.item(), 10*pre_cost.item(), 100*classify_loss.item() ))

			train_loss = loss.item()

		self.eval()

		print("Optimization Finished!")

		return train_loss


	def fit_model(self, rna_data1 = None, adj_orig1 = None, adj_train1 = None, test_E1 = None, test_E_F1 = None,
				  rna_data2       = None, adj_orig2 = None, adj_train2 = None, test_E2 = None, test_E_F2 = None,
				  rna_data_all_1  = None, adj_orig_all_1 = None, rna_data_all_2 = None, adj_orig_all_2 = None,
				  class_label_all = None, used_int = None):

		
		mu_robust       = lamda        = None
		view1_loss      = view2_loss   = 0
		cross_loss      = context_loss = 0 

		rna_data_exp1   = torch.FloatTensor(rna_data1).cuda()
		rna_data_exp2   = torch.FloatTensor(rna_data2).cuda()

		adj_org_norm1   = preprocess_graph(adj_train1).cuda()
		adj_org_norm2   = preprocess_graph(adj_train2).cuda()

		iter            = 0
		train_loss_list = []

		while iter < (10 + 1):

			print(str(iter) + "-----------------")

			print( "HSG by visual features" )

			view1_loss = self.view1.fit( rna_data_exp1, adj_orig1, adj_train1, test_E1, test_E_F1, 
										 lamda    = lamda,           robust_rep = mu_robust,
										 lr_model = self.args.lr_T1, true_class = self.class_label, 
										 training_info = self.training_int,      loc = 0 )

			adj_norm_all_1, _, _, _   = self.view1.graph_processing( adj_orig_all_1 )
			_, mu_all_1, _, class_pre = self.view1.inference(torch.FloatTensor(rna_data_all_1).cuda(), adj_norm_all_1)
			#self.view1.evaluation_metrics(mu_all_1, used_int, class_label_all)
			#self.view1.evaluation_classification(class_label_all, class_pre, used_int)

			print( "SLG by spatial location" )

			view2_loss = self.view2.fit( rna_data_exp2, adj_orig2, adj_train2, test_E2, test_E_F2, 
										 lamda    = lamda,           robust_rep = mu_robust,
										 lr_model = self.args.lr_T2, true_class = self.class_label, 
										 training_info = self.training_int,      loc = 1 )

			adj_norm_all_2, _, _, _    = self.view2.graph_processing( adj_orig_all_2 )
			_, mu_all_2, _, class_pre  = self.view2.inference(torch.FloatTensor(rna_data_all_2).cuda(), adj_norm_all_2)
			#self.view2.evaluation_metrics(mu_all_2, used_int, class_label_all)
			#self.view2.evaluation_classification(class_label_all, class_pre, used_int)

			_, mu_robust_a, class_prediction = self.crossview( mu_all_1, mu_all_2 )

			self.evaluation_metrics( mu_robust_a, class_label_all, used_int)
			#self.evaluation_classification( class_label_all, class_prediction, used_int)

			_, mu1, _, _   = self.view1( rna_data_exp1, adj_org_norm1 )
			_, mu2, _, _   = self.view2( rna_data_exp2, adj_org_norm2 )

			lamda, mu_robust, class_prediction = self.crossview( mu1, mu2 )

			context_loss = self.fit_model_collaborative(rna_data1, adj_train1, rna_data2, adj_train2, 
														lamda.detach(), mu_robust.detach(), self.args.lr_T3, 
														self.training_int, self.class_label, "context" )

			_, mu1, _, _   = self.view1( rna_data_exp1, adj_org_norm1 )
			_, mu2, _, _   = self.view2( rna_data_exp2, adj_org_norm2 )

			lamda, mu_robust, class_prediction = self.crossview( mu1, mu2 )

			if mu_robust is not None and lamda is not None:
				self.evaluation_metrics( mu_robust, self.class_label, self.training_int )
				#_, mu_robust_a, class_prediction = self.learn_robust_representation( rna_data_all_1, adj_orig_all_1, 
				#																	 rna_data_all_2, adj_orig_all_2 )

				#self.evaluation_metrics( mu_robust_a, class_label_all, used_int)
				#self.evaluation_classification( class_label_all, class_prediction, used_int)

			train_loss_list.append( context_loss  )

			if len(train_loss_list) >= 2 :

				print( abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] )

				if abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] < 0.01 :

					print( abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] )
					print( "converged!!!" )
					print( iter )
					break

			iter = iter + 1

	def evaluation_metrics(self, repren, true_class = None, training = None):

		kmeans     = KMeans( n_clusters = self.args.cluster_pre, n_init = 5, random_state = 200 )
		ARI_score, NMI_score = -100, -100

		if true_class is not None and training is not None:
			pred_z1    = kmeans.fit_predict( repren[training,:].data.cpu().numpy() )
			NMI_score  = round( normalized_mutual_info_score( true_class[training].tolist(), pred_z1,  average_method='max' ), 3 )
			ARI_score  = round( metrics.adjusted_rand_score( true_class[training].tolist(), pred_z1 ), 3 )

		elif true_class is not None and training is None:
			pred_z1    = kmeans.fit_predict( repren.data.cpu().numpy() )
			NMI_score  = round( normalized_mutual_info_score( true_class.tolist(), 
															  pred_z1,  average_method='max' ), 3 )
			ARI_score  = round( metrics.adjusted_rand_score( true_class.tolist(), pred_z1 ), 3 )

		print('collaborative clustering ARI score: ' + str(ARI_score) + ' NMI score: ' + str(NMI_score) )
		return ARI_score, NMI_score


	def evaluation_classification(self, true, predict, training = None):

		ARI_score, NMI_score = -100, -100

		prediction    = predict.data.cpu().numpy().tolist()
		predict_class = np.array(list(map(lambda x: x.index(max(x)), prediction)))

		if training is not None:
			ARI_score     = round(metrics.adjusted_rand_score( true[training].tolist(), predict_class[training].tolist() ),3)
			NMI_score     = round(normalized_mutual_info_score( true[training].tolist(), predict_class[training].tolist() ),3)

		else:
			ARI_score     = round(metrics.adjusted_rand_score( true.tolist(), predict_class.tolist() ),3)
			NMI_score     = round(normalized_mutual_info_score( true.tolist(), predict_class.tolist() ),3)

		print( 'collaborative classification ARI score: ' + str(ARI_score) + ' NMI score: ' + str(NMI_score) )
		return ARI_score, NMI_score


	def learn_robust_representation(self, rna_data1, adj_orig1, rna_data2, adj_orig2):

		rna_data_exp1  = torch.FloatTensor(rna_data1).cuda()
		rna_data_exp2  = torch.FloatTensor(rna_data2).cuda()

		adj_org_norm1  = preprocess_graph(adj_orig1).cuda()
		adj_org_norm2  = preprocess_graph(adj_orig2).cuda()

		_, mu1, _, _   = self.view1( rna_data_exp1, adj_org_norm1 )
		_, mu2, _, _   = self.view2( rna_data_exp2, adj_org_norm2 )

		lamda, mu_robust, class_prediction = self.crossview( mu1, mu2 )

		return lamda, mu_robust, class_prediction


	def forward(self, rna_data1, adj_orig1, rna_data2, adj_orig2 ):

		lamda, mu_robust, class_prediction = self.learn_robust_representation( rna_data1, adj_orig1, 
																			   rna_data2, adj_orig2 )
		
		return lamda, mu_robust, class_prediction

class simCLR_model(Module):
	def __init__(self, feature_dim=128):
		super(simCLR_model, self).__init__()

		self.f = []

		#load resnet50 structure
		for name, module in models.resnet50().named_children():
			if name == 'conv1':
				module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
			if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
				self.f.append(module)
		# encoder
		self.f = nn.Sequential(*self.f)
		# projection head
		self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
							   nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

	def forward(self, x):
		x       = self.f(x)
		feature = torch.flatten(x, start_dim=1)
		out     = self.g(feature)

		return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class resnet50_model(Module):
	def __init__(self):
		super(resnet50_model, self).__init__()

		### load pretrained resnet50 model
		resnet50 = models.resnet50(pretrained=True)

		for param in resnet50.parameters():
			param.requires_grad = False

		self.f = []

		for name, module in resnet50.named_children():
			if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
				self.f.append(module)
		# encoder
		self.f = nn.Sequential(*self.f)

	def forward(self, x):
		x       = self.f(x)
		feature = torch.flatten(x, start_dim=1)
	   
		return F.normalize(feature, dim=-1)


class AE(Module):
	def __init__( self, layer_e, layer_d, hidden1, args, droprate = 0.1, type = "NB" ):
		super(AE, self).__init__()
		
		### function definition
		self.encoder     = build_multi_layers( layer_e )

		if type == "NB":
			self.decoder = Decoder_logNorm_NB( layer_d, hidden1, layer_e[0], droprate = droprate )

		else: #Gaussian
			self.decoder = Decoder( layer_d, hidden1, layer_e[0], Type = type, droprate = droprate)

		self.args      = args
		self.type      = type
	
	def inference(self, X = None, scale_factor = 1.0):
		
		latent = self.encoder( X )
		
		### decoder
		if self.type == "NB":
			output        =  self.decoder( latent, scale_factor )
			norm_x        =  output["normalized"]
			disper_x      =  output["disperation"]
			recon_x       =  output["scale_x"]

		else:
			recons_x      =  self.decoder( latent )
			recon_x       =  recons_x
			norm_x        =  None
			disper_x      =  None

		return dict( norm_x   = norm_x, disper_x   = disper_x, 
					 recon_x  = recon_x, latent_z  = latent )


	def return_loss(self, X = None, X_raw = None, scale_factor = 1.0 ):

		output           =  self.inference( X, scale_factor )
		recon_x          =  output["recon_x"]
		disper_x         =  output["disper_x"]

		if self.type == "NB":
			loss         =  log_nb_positive( X_raw, recon_x, disper_x )

		else:
			loss = mse_loss( X, recon_x )

		return loss

		
	def forward( self, X = None, scale_factor = 1.0 ):

		output =  self.inference( X, scale_factor )

		return output


	def predict(self, dataloader, out='z' ):
		
		output = []

		for batch_idx, ( X, size_factor ) in enumerate(dataloader):

			if self.args.use_cuda:
				X, size_factor = X.cuda(), size_factor.cuda()

			X           = Variable( X )
			size_factor = Variable(size_factor)

			result      = self.inference( X, size_factor)

			if out == 'z': 
				output.append( result["latent_z"].detach().cpu() )

			elif out == 'recon_x':
				output.append( result["recon_x"].detach().cpu().data )

			else:
				output.append( result["norm_x"].detach().cpu().data )

		output = torch.cat(output).numpy()
		return output


	def fit( self, train_loader, test_loader ):

		params    = filter(lambda p: p.requires_grad, self.parameters())
		optimizer = optim.Adam( params, lr = self.args.lr_AET, weight_decay = self.args.weight_decay, eps = self.args.eps )

		train_loss_list   = []
		reco_epoch_test   = 0
		test_like_max     = 1000000000
		flag_break        = 0

		patience_epoch         = 0
		self.args.anneal_epoch = 10

		start = time.time()

		for epoch in range( 1, self.args.max_epoch_T + 1 ):

			self.train()
			optimizer.zero_grad()

			patience_epoch += 1

			kl_weight      =  min( 1, epoch / self.args.anneal_epoch )
			epoch_lr       =  adjust_learning_rate( self.args.lr_AET, optimizer, epoch, self.args.lr_AET_F, 10 )

			for batch_idx, ( X, X_raw, size_factor ) in enumerate(train_loader):

				if self.args.use_cuda:
					X, X_raw, size_factor = X.cuda(), X_raw.cuda(), size_factor.cuda()
				
				X, X_raw, size_factor     = Variable( X ), Variable( X_raw ), Variable( size_factor )
				loss1  = self.return_loss( X, X_raw, size_factor )
				loss   = torch.mean( loss1  )

				loss.backward()
				optimizer.step()

			if epoch % self.args.epoch_per_test == 0 and epoch > 0: 
				self.eval()

				with torch.no_grad():

					for batch_idx, ( X, X_raw, size_factor ) in enumerate(test_loader): 

						if self.args.use_cuda:
							X, X_raw, size_factor = X.cuda(), X_raw.cuda(), size_factor.cuda()

						X, X_raw, size_factor     = Variable( X ), Variable( X_raw ), Variable( size_factor )

						loss      = self.return_loss( X, X_raw, size_factor )
						test_loss = torch.mean( loss )

						train_loss_list.append( test_loss.item() )

						print( test_loss.item() )

						if math.isnan(test_loss.item()):
							flag_break = 1
							break

						if test_like_max >  test_loss.item():
							test_like_max   = test_loss.item()
							reco_epoch_test = epoch
							patience_epoch  = 0        

			if flag_break == 1:
				print("containin NA")
				print(epoch)
				break

			if patience_epoch >= 30 :
				print("patient with 50")
				print(epoch)
				break
			
			if len(train_loss_list) >= 2 :
				if abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] < 1e-4 :

					print( abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] )
					print( "converged!!!" )
					print( epoch )
					break

		duration = time.time() - start

		print('Finish training, total time is: ' + str(duration) + 's' )
		self.eval()
		print(self.training)

		print( 'train likelihood is :  '+ str(test_like_max) + ' epoch: ' + str(reco_epoch_test) )
