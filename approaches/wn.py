# from cProfile import label
import sys, time, os
import math

import numpy as np
# from pytest import param
# from sqlalchemy import false
# from sympy import arg
import torch
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
import kornia as K

import time
import csv
from utils import *
import networks.wn_net as network
from layers.wn_layer import _WeightNormLayer, WeightNormLinear, WeightNormConv2D
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.utils import shuffle
from torch.utils.data import  TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
from arguments import get_args
args = get_args()

class Appr(object):

    def __init__(self, input_size, output_size, args):
        Net = getattr(network, args.arch)
        self.model = Net(input_size=input_size, output_size=output_size, norm_type=args.norm_type).to(device)
        print(self.model)
        self.nepochs = args.nepochs
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.lr = args.lr
        self.lr_rho = args.lr_rho
        self.lr_min = self.lr/100
        self.lr_factor = args.lr_factor
        self.lr_patience = args.lr_patience 
        self.clipgrad = 100
        self.optim = args.optimizer
        self.tasknum = args.tasknum
        self.fix = args.fix
        self.experiment = args.experiment
        self.approach = args.approach
        self.arch = args.arch
        self.seed = args.seed
        self.norm_type = args.norm_type
        self.ablation = args.ablation
        self.train_size = args.train_size
        self.logger = None
        self.thres = args.thres
        self.prune_method = args.prune_method

        self.args = args
        self.lamb = args.lamb
        self.check_point = None
        
        self.ce = torch.nn.CrossEntropyLoss()

        self.get_name()
        self.best_path = []

    def get_name(self):
        self.log_name = '{}_{}_{}_{}_{}_{}_lr_{}_batch_{}_epoch_{}_optim_{}_train_{}'.format(
                                        self.experiment, self.approach, self.ablation, self.arch, self.args.norm_type, 
                                        self.seed, self.lr, self.batch_size, self.nepochs, self.optim, self.train_size)
        
    def resume(self):
        try:
            self.get_name()
            self.check_point = torch.load(f'../result_data/trained_model/{self.log_name}.model')
            self.model = self.check_point['model']
        except:
            self.check_point = None

    def count_params(self):
        return self.model.count_params()[0]

    def _get_optimizer(self, lr=None):
        if lr is None: lr=self.lr

        params = self.model.parameters()

        if self.optim == 'SGD':
            optimizer = torch.optim.SGD(params, lr=lr,
                          weight_decay=0.0, momentum=0.9)
        elif self.optim == 'Adam':
            optimizer = torch.optim.Adam(params, lr=lr)

        return optimizer

    def train(self, train_loader, valid_loader, train_transform, valid_transform):

        if self.check_point is None:
            print('Start training')
            self.model = self.model.to(device)
            self.check_point = {'model':self.model, 'squeeze':True, 'optimizer':self._get_optimizer(), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}
            self.get_name()
            torch.save(self.check_point, f'../result_data/trained_model/{self.log_name}.model')
        else: 
            print('Continue training')

        self.model = self.model.to(device)
        self.mean = train_loader.dataset.tensors[0].mean(dim=(0, 2, 3))
        var = train_loader.dataset.tensors[0].var(dim=(0, 2, 3))
        next_ks = self.model.WN[0].ks
        self.std = (var.sum() * next_ks) ** 0.5
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

        # train_transform = torch.nn.Sequential(
        #     K.augmentation.Normalize(mean, std),
        # )
        # valid_transform = torch.nn.Sequential(
        #     K.augmentation.Normalize(mean, std),
        # )
        print(self.log_name)

        count = 0
        for p in self.model.parameters():
            count += p.numel()
        print('num params:', count)
        train_loss,train_acc=self.eval(train_loader,valid_transform)
        print('| Train: loss={:.3f}, acc={:5.2f}% |'.format(train_loss,100*train_acc), end='')

        valid_loss,valid_acc=self.eval(valid_loader,valid_transform)
        print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))

        lr = self.check_point['lr']
        patience = self.check_point['patience']
        self.optimizer = self.check_point['optimizer']
        start_epoch = self.check_point['epoch'] + 1
        squeeze = self.check_point['squeeze']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.nepochs)
        train_accs = []
        valid_accs = []
        if squeeze:
            best_acc = train_acc
        else:
            best_acc = valid_acc

        try:
            for e in range(start_epoch, self.nepochs):
                clock0=time.time()
                self.train_epoch(train_loader, train_transform)
            
                clock1=time.time()
                train_loss,train_acc=self.eval(train_loader, valid_transform)
                clock2=time.time()
                print('| Epoch {:2d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}% |'.format(
                    e+1,1000*(clock1-clock0),
                    1000*(clock2-clock1),train_loss,100*train_acc),end='')

                valid_loss,valid_acc=self.eval(valid_loader, valid_transform)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc),end='')
                scheduler.step()
                # Adapt lr
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    self.check_point = {'model':self.model, 'optimizer':self.optimizer, 'squeeze':squeeze, 'epoch':e, 'lr':lr, 'patience':patience}
                    torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))
                    patience = self.lr_patience
                    print(' *', end='')
                # else:
                #     if e > 0:
                #         patience -= 1
                #         if patience <= 0:
                #             lr /= self.lr_factor
                #             print(' lr={:.1e}'.format(lr), end='')
                #             if lr < self.lr_min:
                #                 print()
                #                 break
                                
                #             patience = self.lr_patience
                #             self.optimizer = self._get_optimizer(lr)

                print()
                train_accs.append(train_acc)
                valid_accs.append(valid_acc)
                # if self.logger is not None:
                #     self.logger.log_metrics({
                #         'train acc':train_acc,
                #         'valid acc':valid_acc
                #     }, epoch=e)

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
            self.model = self.check_point['model']

        self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
        self.model = self.check_point['model']
        print(train_accs)
        print(valid_accs)

    def train_batch(self, images, targets):
        outputs = self.model.forward(images)
        loss = self.ce(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step()
        if 'normalize' not in self.ablation:
            self.model.normalize()

    def eval_batch(self,images, targets):
        outputs = self.model.forward(images)
        loss=self.ce(outputs,targets)
        values,indices=outputs.max(1)
        hits=(indices==targets).float()
        return loss.data.cpu().numpy()*len(targets), hits.sum().data.cpu().numpy()

    def train_epoch(self, data_loader, train_transform):
        self.model.train()
        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            if train_transform:
                images = train_transform(images)
            # images = (images - self.mean.view(1, -1, 1, 1)) / self.std
                            
            self.train_batch(images, targets)
        s_H = 1
        for m in self.model.modules():
            if isinstance(m, _WeightNormLayer):
                s_H *= m.weight.norm(2).item()
        print('s_H:', round(s_H, 3))

    def eval(self,data_loader, valid_transform):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        for images, targets in data_loader:
            images=images.to(device)
            targets=targets.to(device)
            # images = (images - self.mean.view(1, -1, 1, 1)) / self.std
            if valid_transform:
                images = valid_transform(images)
                    
            loss, hits = self.eval_batch(images, targets)
            total_loss += loss
            total_acc += hits
            total_num += len(targets)
                
        return total_loss/total_num,total_acc/total_num