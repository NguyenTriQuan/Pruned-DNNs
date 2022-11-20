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
    input_size=[1,28,28]
    output_size = 10

    mean = (0.1307)
    std = (0.3081)
    
    train_set = datasets.MNIST('../dat/', train=True, download=True)
    test_set = datasets.MNIST('../dat/', train=False, download=True)

    train_data, train_targets = train_set.data.float(), train_set.targets.long()
    test_data, test_targets = test_set.data.float(), test_set.targets.long()

    train_data = train_data.unsqueeze(1)/255.0
    test_data = test_data.unsqueeze(1)/255.0
    
    r=np.arange(train_data.size(0))
    r=np.array(shuffle(r,random_state=args.seed),dtype=int)
    pivot=int(args.train_size*len(r))
    idx=torch.LongTensor(r[:pivot])

    train_loader = DataLoader(TensorDataset(train_data[idx], train_targets[idx]), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_targets), batch_size=args.val_batch_size, shuffle=False)

    if args.augment:
        train_transform = torch.nn.Sequential(
            K.augmentation.RandomResizedCrop(size=(28, 28), scale=(0.2, 1.0), same_on_batch=False),
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