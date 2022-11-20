import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torch.utils.data import  TensorDataset, DataLoader

import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle
from torch.utils.data import  TensorDataset, DataLoader
import kornia as K

def get(args, pc_valid=0.0):
    input_size=[3,32,32]
    output_size = 10

    mean=torch.tensor([x/255 for x in [125.3,123.0,113.9]])
    std=torch.tensor([x/255 for x in [63.0,62.1,66.7]])
    
    train_set = datasets.CIFAR10('../dat/',train=True,download=True)
    test_set = datasets.CIFAR10('../dat/',train=False,download=True)

    train_data, train_targets = torch.FloatTensor(train_set.data), torch.LongTensor(train_set.targets)
    test_data, test_targets = torch.FloatTensor(test_set.data), torch.LongTensor(test_set.targets)
    
    train_data = train_data.permute(0, 3, 1, 2)/255.0
    test_data = test_data.permute(0, 3, 1, 2)/255.0
    
    r=np.arange(train_data.size(0))
    r=np.array(shuffle(r,random_state=args.seed),dtype=int)
    pivot=int(args.train_size*len(r))
    idx=torch.LongTensor(r[:pivot])

    train_loader = DataLoader(TensorDataset(train_data[idx], train_targets[idx]), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_targets), batch_size=args.val_batch_size, shuffle=False)

    if args.augment:
        train_transform = torch.nn.Sequential(
            K.augmentation.RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), same_on_batch=False),
            K.augmentation.RandomHorizontalFlip(),
            K.augmentation.Normalize(mean, std),
        )
    else:
        train_transform = torch.nn.Sequential(
            K.augmentation.Normalize(mean, std),
        )
        
    valid_transform = torch.nn.Sequential(
        K.augmentation.Normalize(mean, std),
    )
    return train_loader, test_loader, train_transform, valid_transform, input_size, output_size