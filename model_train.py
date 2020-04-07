import argparse
import sys
import os
import shutil
import json
import time
import warnings
from random import sample

import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR

from cgcnn.cgcnn_hd import CrystalGraphConvNet
from cgcnn.data import collate_pool, get_train_val_test_loader, CIFData

def main():
	#taken from sys.argv
	chk_name = sys.argv[1]
	best_name = sys.argv[2]
	save_name = sys.argv[3]

	#var. for dataset loader
	root_dir = '/your/model/path/'
	max_num_nbr = 8
	radius = 4
	dmin = 0
	step = 0.2
	random_seed = 1234
	batch_size = 64
	N_tot = len(open(root_dir+'/id_prop.csv').readlines())
	N_tr = int(N_tot*0.8)
	N_val = int(N_tot*0.1)
	N_test = N_tot - N_tr - N_val

	train_idx = list(range(N_tr))
	val_idx = list(range(N_tr,N_tr+N_val))
	test_idx = list(range(N_tr+N_val,N_tr+N_val+N_test))

	num_workers = 0
	pin_memory = True
	return_test = True

	#var for model
#	atom_fea_len,h_fea_len,n_conv,n_h,lr_decay_rate = Hyp_loader(root_dir,hyp_idx)
	atom_fea_len = 90
	h_fea_len = 2*atom_fea_len
	n_conv = 5
	n_h = 2
	lr_decay_rate = 0.97

	lr = 0.001
	weight_decay = 0.0
	resume = False
	resume_path = 'ddd'

	#var for training
	best_mae_error = 1e10
	start_epoch = 0
	epochs = 200

	#setup
	dataset = CIFData(root_dir,max_num_nbr,radius,dmin,step,random_seed)
	collate_fn = collate_pool

	train_loader, val_loader, test_loader = get_train_val_test_loader(dataset,collate_fn,batch_size,
																					train_idx,val_idx,test_idx,num_workers,pin_memory,return_test)

	sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]
	_, sample_target, _ = collate_pool(sample_data_list)
	normalizer = Normalizer(sample_target)

	#build model
	structures, _, _ = dataset[0]
	orig_atom_fea_len = structures[0].shape[-1]
	nbr_fea_len = structures[1].shape[-1]
	model = CrystalGraphConvNet(orig_atom_fea_len,nbr_fea_len,atom_fea_len,n_conv,h_fea_len,n_h)
	model.cuda()

	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(),lr,weight_decay=weight_decay)
	scheduler = ExponentialLR(optimizer, gamma=lr_decay_rate)
	
	# optionally resume from a checkpoint
	if resume:
		print("=> loading checkpoint '{}'".format(args.resume))
		start_epoch = checkpoint['epoch']
		best_mae_error = checkpoint['best_mae_error']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		normalizer.load_state_dict(checkpoint['normalizer'])
		print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

	t0 = time.time()
	for epoch in range(start_epoch,epochs):
		train(train_loader,model,criterion,optimizer,epoch,normalizer)
		mae_error = validate(val_loader,model,criterion,normalizer)

		scheduler.step()
		is_best = mae_error < best_mae_error
		best_mae_error = min(mae_error, best_mae_error)
		save_checkpoint({'epoch': epoch,
            				 'state_dict': model.state_dict(),
										 'best_mae_error': best_mae_error,
                     'optimizer': optimizer.state_dict(),
                     'normalizer': normalizer.state_dict()},is_best,chk_name,best_name)
	t1 = time.time()
	print('--------Training time in sec-------------')
	print(t1-t0)
	print('---------Best Model on Validation Set---------------')
	best_checkpoint = torch.load(best_name)
	print(best_checkpoint['best_mae_error'].cpu().numpy())
	print('---------Evaluate Model on Test Set---------------')
	model.load_state_dict(best_checkpoint['state_dict'])
	validate(test_loader, model, criterion, normalizer, test=True, save_name=save_name)

def train(train_loader, model, criterion, optimizer, epoch, normalizer):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	mae_errors = AverageMeter()
    
	#switch to train model
	model.train()
	end = time.time()
	for i, (input,target,_) in enumerate(train_loader):
		data_time.update(time.time() - end)
		input_var = (Variable(input[0].cuda(async=True)),
								 Variable(input[1].cuda(async=True)),
								 input[2].cuda(async=True),
								 [crys_idx.cuda(async=True) for crys_idx in input[3]])

		#normalize target
		target_normed = normalizer.norm(target)
		target_var = Variable(target_normed.cuda(async=True))
		
		#compute output
		output,_ = model(*input_var)
		loss = criterion(output, target_var)

		#measure accuracy
		mae_error = mae(normalizer.denorm(output.data.cpu()), target)
		losses.update(loss.data.cpu()[0], target.size(0))
		mae_errors.update(mae_error, target.size(0))

		#backward operation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
		#measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
          epoch, i, len(train_loader), batch_time=batch_time,
          data_time=data_time, loss=losses, mae_errors=mae_errors))

def validate(val_loader,model,criterion,normalizer,test=False,save_name='test.csv'):
	batch_time = AverageMeter()
	losses = AverageMeter()
	mae_errors = AverageMeter()

	if test:
		test_targets = []
		test_preds = []
		test_cif_ids = []

	#switch to evaluate mode
	model.eval()

	end = time.time()
	for i, (input, target, batch_cif_ids) in enumerate(val_loader):
		input_var = (Variable(input[0].cuda(async=True), volatile=True),
								 Variable(input[1].cuda(async=True), volatile=True),
								 input[2].cuda(async=True),
								 [crys_idx.cuda(async=True) for crys_idx in input[3]])

		target_normed = normalizer.norm(target)
		target_var = Variable(target_normed.cuda(async=True),volatile=True)

		#compute output
		output, _ = model(*input_var)
		loss = criterion(output, target_var)

		#measure accuracy and record loss
		mae_error = mae(normalizer.denorm(output.data.cpu()), target)
		losses.update(loss.data.cpu()[0], target.size(0))
		mae_errors.update(mae_error, target.size(0))
		if test:
			test_pred = normalizer.denorm(output.data.cpu())
			test_target = target
			test_preds += test_pred.view(-1).tolist()
			test_targets += test_target.view(-1).tolist()
			test_cif_ids += batch_cif_ids

		#measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		print('Test: [{0}/{1}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
          i, len(val_loader), batch_time=batch_time, loss=losses,
          mae_errors=mae_errors))

	if test:
		star_label = '**'
		import csv
		with open(save_name, 'w') as f:
			writer = csv.writer(f)
			for cif_id, target, pred in zip(test_cif_ids,test_targets,test_preds):
				writer.writerow((cif_id, target, pred))
	else:
		star_label = '*'

	print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,mae_errors=mae_errors))
	return mae_errors.avg

class Normalizer(object):
	def __init__(self, tensor):
		self.mean = torch.mean(tensor)
		self.std = torch.std(tensor)

	def norm(self, tensor):
		return (tensor - self.mean) / self.std

	def denorm(self, normed_tensor):
		return normed_tensor * self.std + self.mean

	def state_dict(self):
		return {'mean': self.mean,'std': self.std}

	def load_state_dict(self, state_dict):
		self.mean = state_dict['mean']
		self.std = state_dict['std']

def mae(prediction, target):
	return torch.mean(torch.abs(target - prediction))

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self,val,n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def save_checkpoint(state,is_best,chk_name,best_name):
	torch.save(state, chk_name)
	if is_best:
		shutil.copyfile(chk_name,best_name)

if __name__ == '__main__':
	main()
