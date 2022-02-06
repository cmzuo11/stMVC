# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:17:25 2021

@author: chunman zuo
"""

import numpy as np
import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
import time

from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm

from stMVC.loss_function import supervised_multiple_loss_function


def build_multi_layers(layers, use_batch_norm=True, dropout_rate = 0.1 ):
	"""Build multilayer linear perceptron"""

	if dropout_rate > 0:
		fc_layers = nn.Sequential(
			collections.OrderedDict(
				[
					(
						"Layer {}".format(i),
						nn.Sequential(
							nn.Linear(n_in, n_out),
							nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
							nn.ReLU(),
							nn.Dropout(p=dropout_rate),
						),
					)

					for i, (n_in, n_out) in enumerate( zip(layers[:-1], layers[1:] ) )
				]
			)
		)

	else:
		fc_layers = nn.Sequential(
			collections.OrderedDict(
				[
					(
						"Layer {}".format(i),
						nn.Sequential(
							nn.Linear(n_in, n_out),
							nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
							nn.ReLU(),
						),
					)

					for i, (n_in, n_out) in enumerate( zip(layers[:-1], layers[1:] ) )
				]
			)
		)
		
		
	
	return fc_layers


class Encoder(Module):
	
	## for one modulity
	def __init__(self, layer, hidden, Z_DIMS, droprate = 0.1 ):
		super(Encoder, self).__init__()
		
		if len(layer) > 1:
			self.fc1   =  build_multi_layers( layers = layer, dropout_rate = droprate )
			
		self.layer = layer
		self.fc_means   =  nn.Linear(hidden, Z_DIMS)
		self.fc_logvar  =  nn.Linear(hidden, Z_DIMS)
		
	def reparametrize(self, means, logvar):

		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mul(std).add_(means)

		else:
		  return means

	def return_all_params(self, x):

		if len(self.layer) > 1:
			h = self.fc1(x)
		else:
			h = x

		mean_x   = self.fc_means(h)
		logvar_x = self.fc_logvar(h)
		latent   = self.reparametrize(mean_x, logvar_x)
		
		return mean_x, logvar_x, latent, h

		
	def forward(self, x):

		_, _, latent = self.return_all_params( x )
		
		return latent


class Decoder(Module):
	### for scATAC-seq
	def __init__(self, layer, hidden, input_size, Type = "Bernoulli" , droprate = 0.1 ):
		super(Decoder, self).__init__()
		
		if len(layer) >1 :
			self.decoder   =  build_multi_layers( layer, dropout_rate = droprate )
		
		self.decoder_x = nn.Linear( hidden, input_size )
		self.Type      = Type
		self.layer     = layer

	def forward(self, z):
		
		if len(self.layer) >1 :
			latent  = self.decoder( z )
		else:
			latent = z
			
		recon_x = self.decoder_x( latent )
		
		if self.Type == "Bernoulli":
			Final_x = torch.sigmoid(recon_x)
			
		elif self.Type == "Gaussian":
			Final_x = F.relu(recon_x)

		else:
			Final_x = recon_x
		
		return Final_x



class Decoder_logNorm_NB(Module):
	
	### for scRNA-seq
	
	def __init__(self, layer, hidden, input_size, droprate = 0.1  ):
		
		super(Decoder_logNorm_NB, self).__init__()

		self.decoder =  build_multi_layers( layers = layer, dropout_rate = droprate  )
		
		self.decoder_scale = nn.Linear(hidden, input_size)
		self.decoder_r = nn.Linear(hidden, input_size)

	def forward(self, z, scale_factor = torch.tensor(1.0)):
		
		latent = self.decoder(z)
		
		normalized_x = F.softmax( self.decoder_scale( latent ), dim = 1 )  ## mean gamma

		batch_size   = normalized_x.size(0)
		scale_factor.resize_(batch_size,1)
		scale_factor.repeat(1, normalized_x.size(1))

		scale_x      =  torch.exp(scale_factor) * normalized_x
		
		disper_x     =  torch.exp( self.decoder_r( latent ) ) ### theta
		
		return dict( normalized      =  normalized_x,
					 disperation     =  disper_x,
					 scale_x         =  scale_x,
				   )

class GraphAttentionLayer(Module):
	"""
	Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
	"""
	def __init__(self, in_features, out_features, dropout, alpha = 0.2, concat=True):
		super(GraphAttentionLayer, self).__init__()
		self.dropout      = dropout
		self.in_features  = in_features
		self.out_features = out_features
		self.alpha        = alpha
		self.concat       = concat

		self.W         = nn.Parameter(torch.empty(size=(in_features, out_features)))
		nn.init.xavier_uniform_(self.W.data, gain=1.414)

		self.a         = nn.Parameter(torch.empty(size=(2*out_features, 1)))
		nn.init.xavier_uniform_(self.a.data, gain=1.414)

		self.leakyrelu = nn.LeakyReLU(self.alpha)

	def forward(self, h, adj):
		Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
		e  = self._prepare_attentional_mechanism_input(Wh)

		zero_vec  = -9e15*torch.ones_like(e)
		attention = torch.where(adj.to_dense() > 0, e, zero_vec)
		attention = F.softmax(attention, dim=1)
		attention = F.dropout(attention, self.dropout, training=self.training)
		h_prime   = torch.matmul(attention, Wh)

		if self.concat:
			return F.elu(h_prime)
		else:
			return h_prime

	def _prepare_attentional_mechanism_input(self, Wh):
		# Wh.shape (N, out_feature)
		# self.a.shape (2 * out_feature, 1)
		# Wh1&2.shape (N, 1)
		# e.shape (N, N)
		Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
		Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
		# broadcast add
		e   = Wh1 + Wh2.T
		return self.leakyrelu(e)

	def __repr__(self):
		return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class CrossViewAttentionLayer(Module):

	def __init__(self, NFeature, nClass, dropout, alpha = 0.2):
		super(CrossViewAttentionLayer, self).__init__()

		self.dropout   = dropout
		self.NFeature  = NFeature
		self.alpha     = alpha

		self.a1        = nn.Parameter(torch.empty(size=(2*NFeature, 1)))
		self.a2        = nn.Parameter(torch.empty(size=(2*NFeature, 1)))

		self.classi1   = nn.Parameter(torch.empty(size=(NFeature, 2*NFeature)))
		self.classi2   = nn.Parameter(torch.empty(size=(2*NFeature, nClass)))

		nn.init.xavier_uniform_(self.a1.data, gain=1.414)
		nn.init.xavier_uniform_(self.a2.data, gain=1.414)

		nn.init.xavier_uniform_(self.classi1.data, gain=1.414)
		nn.init.xavier_uniform_(self.classi2.data, gain=1.414)

		self.leakyrelu = nn.LeakyReLU( alpha )

	def inference( self, h1 = None, h2 = None):

		e         = self._prepare_attentional_mechanism_input(h1, h2)
		lamda     = F.softmax(e, dim=1)
		lamda     = F.dropout(lamda, self.dropout, training=self.training)

		h_prime1  = lamda[:,0].repeat(self.NFeature,1).T * h1
		h_prime2  = lamda[:,1].repeat(self.NFeature,1).T * h2

		h_robust  = h_prime1 + h_prime2

		Wh        = torch.mm( h_robust, self.classi1 )
		Wh        = torch.mm( F.relu(Wh), self.classi2 )

		return lamda, h_robust, F.softmax(Wh, dim=1)


	def forward(self, h1 = None, h2 = None):

		lamda, h_robust, class_prediction = self.inference(h1, h2)

		return lamda, h_robust, class_prediction
			

	def _prepare_attentional_mechanism_input(self, h1 = None, h2 = None):
		# h1 & h2.shape (N, out_feature)
		# a1 & a2.shape (2 * out_feature, 1)
		# Wh1&2.shape (N, 1)
		# e.shape (N, 2)
		Wh1 = torch.matmul(torch.cat((h1, h2), 1), self.a1)
		Wh2 = torch.matmul(torch.cat((h1, h2), 1), self.a2)
		# broadcast add
		e   = torch.cat((Wh1, Wh2), 1)

		return self.leakyrelu(e)

class Integrate_multiple_view_model(Module):

	def __init__(self, NFeature, nClass, dropout, alpha = 0.2, type = "Mean"):
		super(Integrate_multiple_view_model, self).__init__()

		## by mean

		self.W1        = nn.Parameter(torch.empty(size=(NFeature, 2*NFeature)))
		self.W2        = nn.Parameter(torch.empty(size=(2*NFeature, nClass)))

		nn.init.xavier_uniform_(self.W1.data, gain=1.414)
		nn.init.xavier_uniform_(self.W2.data, gain=1.414)

		self.type     = type

	def inference_mean( self, h1 = None, h2 = None):

		rep_sum = h1 + h2
		rep     = torch.div( rep_sum, 2.0 )

		Wh      = torch.mm(rep, self.W1)    #classifier1
		Wh      = torch.mm(F.relu(Wh), self.W2)  #classifier1

		return rep, F.softmax(Wh, dim=1)

	def forward(self, h1 = None, h2 = None):

		rep, class_prediction = self.inference_mean( h1, h2 )

		return torch.ones(rep.size(0), 3).cuda(), rep, class_prediction

class GraphConvolution(Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, out_features, dropout=0., act=F.relu):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.dropout = dropout
		self.act = act
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.xavier_uniform_(self.weight)

	def forward(self, input, adj):
		input = F.dropout(input, self.dropout, self.training)
		support = torch.mm(input, self.weight)
		output = torch.spmm(adj, support)
		output = self.act(output)
		return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.out_features) + ')'
	
