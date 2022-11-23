import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import math
import copy
import utils
from utils import *
from arguments import get_args
import importlib
import random
# import comet_ml at the top of your file
# from comet_ml import Experiment, ExistingExperiment
import json
import torch.backends.cudnn as cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('../result_data/trained_model/', exist_ok=True)
os.makedirs('../result_data/logger/', exist_ok=True)
args = get_args()
tstart = time.time()

args.max_params = max(args.max_params, 0)
args.max_mul = max(args.max_mul, 0)

print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)

# Seed
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# dataloader = importlib.import_module('dataloaders.{}'.format(args.experiment))
# data, taskcla, inputsize = dataloader.get(batch_size=args.batch_size, val_batch_size=args.val_batch_size, seed=args.seed, tasknum=args.tasknum)

# try:
dataloader = importlib.import_module('dataloaders.{}'.format(args.experiment))
train_loader, test_loader, train_transform, valid_transform, input_size, output_size = dataloader.get(args)
# except:
#     dataloader = importlib.import_module('dataloaders.single_task')
#     data, taskcla, inputsize = dataloader.get(batch_size=args.batch_size, val_batch_size=args.val_batch_size, seed=args.seed, name=args.experiment)

print('Input size =', input_size, '\nOutput size =', output_size)

approach = importlib.import_module('approaches.{}'.format(args.approach))
appr = approach.Appr(input_size, output_size, args)

start_task = args.start_task
if args.resume:
    # with open(f'../result_data/logger/{appr.log_name}.json', 'r') as f:
    #     KEY = json.load(f)

    # appr.logger = ExistingExperiment(
    #     api_key="YSY2PKZaRYWMWkA9XvW0SnJzF",
    #     previous_experiment=KEY
    # )
    start_task = appr.resume()
# else:
#     appr.logger = Experiment(
#         api_key="YSY2PKZaRYWMWkA9XvW0SnJzF",
#         project_name="sccl",
#         workspace="nguyentriquan",
#     )
#     appr.logger.set_name(appr.log_name)
#     with open(f'../result_data/logger/{appr.log_name}.json', 'w') as f:
#         json.dump(appr.logger.get_key(), f)

if device == 'cuda':
    net = torch.nn.DataParallel(appr.model)
    cudnn.benchmark = True

print('-' * 100)
print(f'Train size = {train_loader.dataset.tensors[0].shape[0]} / Test size = {test_loader.dataset.tensors[0].shape[0]}')
appr.train(train_loader, test_loader, train_transform, valid_transform)

test_loss, test_acc = appr.eval(test_loader, valid_transform)
print('>>> Test: loss={:.3f}, acc={:5.2f}% <<<'.format(test_loss, 100 * test_acc))

print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
