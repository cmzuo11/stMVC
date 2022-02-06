# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:17:34 2021

@author: chunman zuo
"""

import torch
import torch.nn.modules.loss
import torch.nn.functional as F
torch.nn.CrossEntropyLoss

def loss_function_total(recover1, labels1, mu1, logvar1, prediction1, norm1, pos_weight1,
						recover2, labels2, mu2, logvar2, prediction2, norm2, pos_weight2,
						recover3, labels3, mu3, logvar3, prediction3, norm3, pos_weight3,
						n_nodes = None,    lamda = None, robust_rep = None, 
						true_class = None, training = None):
		
	cost1 = norm1 * F.binary_cross_entropy_with_logits(recover1, labels1, pos_weight=pos_weight1)
	cost2 = norm2 * F.binary_cross_entropy_with_logits(recover2, labels2, pos_weight=pos_weight2)
	cost3 = norm3 * F.binary_cross_entropy_with_logits(recover3, labels3, pos_weight=pos_weight3)

	if robust_rep is not None and lamda is not None:
		robust_loss1 = torch.sum(lamda[:,0] * ( torch.sum( (mu1 - robust_rep)**2, dim=1) ))
		robust_loss2 = torch.sum(lamda[:,1] * ( torch.sum( (mu2 - robust_rep)**2, dim=1) ))
		robust_loss3 = torch.sum(lamda[:,2] * ( torch.sum( (mu3 - robust_rep)**2, dim=1) ))

	else:
		robust_loss1 = torch.tensor(0.0)
		robust_loss2 = torch.tensor(0.0)
		robust_loss3 = torch.tensor(0.0)

	if logvar1 is None or logvar2 is None or logvar3 is None:
		KLD1 = torch.tensor(0.0)
		KLD2 = torch.tensor(0.0)
		KLD3 = torch.tensor(0.0)
	else:
		KLD1 = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar1 - mu1.pow(2) - logvar1.exp().pow(2), 1))
		KLD2 = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar2 - mu2.pow(2) - logvar2.exp().pow(2), 1))
		KLD3 = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar3 - mu3.pow(2) - logvar3.exp().pow(2), 1))

	if prediction1 is None or prediction2 is None or prediction3 is None:
		predict_loss1 = torch.tensor(0.0)
		predict_loss2 = torch.tensor(0.0)
		predict_loss3 = torch.tensor(0.0)

	else:
		predict_loss1 = supervised_multiple_loss_function(prediction1, true_class, training)
		predict_loss2 = supervised_multiple_loss_function(prediction2, true_class, training)
		predict_loss3 = supervised_multiple_loss_function(prediction3, true_class, training)
		
	return cost1+cost2+cost3, robust_loss1+robust_loss2+robust_loss3, KLD1+KLD2+KLD3, predict_loss1+predict_loss2+predict_loss3

def loss_function_total_2(recover1, labels1, mu1, logvar1, prediction1, norm1, pos_weight1,
						  recover2, labels2, mu2, logvar2, prediction2, norm2, pos_weight2,
						  n_nodes = None,    lamda = None, robust_rep = None, 
						  true_class = None, training = None):
		
	cost1 = norm1 * F.binary_cross_entropy_with_logits(recover1, labels1, pos_weight=pos_weight1)
	cost2 = norm2 * F.binary_cross_entropy_with_logits(recover2, labels2, pos_weight=pos_weight2)

	if robust_rep is not None and lamda is not None:
		robust_loss1 = torch.sum(lamda[:,0] * ( torch.sum( (mu1 - robust_rep)**2, dim=1) ))
		robust_loss2 = torch.sum(lamda[:,1] * ( torch.sum( (mu2 - robust_rep)**2, dim=1) ))

	else:
		robust_loss1 = torch.tensor(0.0)
		robust_loss2 = torch.tensor(0.0)

	if logvar1 is None or logvar2 is None:
		KLD1 = torch.tensor(0.0)
		KLD2 = torch.tensor(0.0)
	else:
		KLD1 = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar1 - mu1.pow(2) - logvar1.exp().pow(2), 1))
		KLD2 = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar2 - mu2.pow(2) - logvar2.exp().pow(2), 1))

	if prediction1 is None or prediction2 is None:
		predict_loss1 = torch.tensor(0.0)
		predict_loss2 = torch.tensor(0.0)

	else:
		predict_loss1 = supervised_multiple_loss_function(prediction1, true_class, training)
		predict_loss2 = supervised_multiple_loss_function(prediction2, true_class, training)
		
	return cost1+cost2, robust_loss1+robust_loss2, KLD1+KLD2, predict_loss1+predict_loss2

def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight, 
				  lamda = None, robust_rep = None, prediction = None, 
				  true_class = None, training = None, loc = 0):
		
	cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

	if robust_rep is not None and lamda is not None:
		lamda       = lamda.detach()
		robust_rep  = robust_rep.detach()

		robust_loss = torch.sum(lamda[:,loc] * ( torch.sum( (mu - robust_rep)**2, dim=1) ))
		#print(robust_loss)

	else:
		robust_loss = torch.tensor(0.0)

	if logvar is None:
		KLD = torch.tensor(0.0)
	else:
		KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

	if prediction is None:
		predict_class = torch.tensor(0.0)

	else:
		predict_class = supervised_multiple_loss_function(prediction, true_class, training)
		
	return cost, KLD, robust_loss, predict_class

def supervised_multiple_loss_function(preds, labels, training = None):

	if training is None:
		labels = torch.from_numpy( labels ).cuda()
		cost   = F.cross_entropy(preds, labels)

	else:
		labelss = torch.from_numpy( labels[training] ).cuda()
		predict = preds[training,:]

		cost    = F.cross_entropy(predict, labelss )
	
	return torch.mean(cost) 

def log_nb_positive(x, mu, theta, eps=1e-8):
	
	x = x.float()
	
	if theta.ndimension() == 1:
		theta = theta.view(
			1, theta.size(0)
		)  # In this case, we reshape theta for broadcasting

	log_theta_mu_eps = torch.log(theta + mu + eps)

	res = (
		theta * (torch.log(theta + eps) - log_theta_mu_eps)
		+ x * (torch.log(mu + eps) - log_theta_mu_eps)
		+ torch.lgamma(x + theta)
		- torch.lgamma(theta)
		- torch.lgamma(x + 1)
	)

	#print(res.size())

	return - torch.sum( res, dim = 1 )

def mse_loss(y_true, y_pred):

	y_pred = y_pred.float()
	y_true = y_true.float()

	ret = torch.pow( (y_pred - y_true) , 2)

	return torch.sum( ret, dim = 1 )
