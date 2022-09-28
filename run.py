#!/usr/bin/env python
# coding: utf-8
#import GPUtil
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import os
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import utils as tvutils
from PIL import Image

from scripts import utils, models
import trainer, tester

# ############################################
#     OPTIONS
# ############################################
parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='Prints debug info', action='store_true')
parser.add_argument('--agave', help='Specify whether running on AGAVE cluser', action='store_true')

parser.add_argument('--ngpu', help='Specify number of GPUs to use', type=int, default=1)
parser.add_argument('--train_gpu', help='Specify which GPU to train on | Options: ["0", "1", "2", ...]', default='0')
parser.add_argument('--test_gpu', help='Specify which GPU to test/val on | Options: ["0", "1", "2", ...]', default='0')
parser.add_argument('--batch_size', help='Specify number of samples in each batch iteration', type=int, default=20)
parser.add_argument('--num_workers', help='Specify num. of workers to load images', type=int, default=0)
parser.add_argument('--checkpoint', help='Determine checkpoint | Options: [-2: PretrainedModel, -1: Latest, 0: Restart, n: epoch(n)', type=int, default=-2)

parser.add_argument('--dataset', help='Specify Dataset | Options: ["Synthetic-COIL20", "diffusercam-berkeley", "diffcam-mini"]', default='diffusercam-berkeley')
parser.add_argument('--method', help='Specify training method type | Options: ["UNet", "GAN", "WGAN"]', default='GAN')
parser.add_argument('--phase', help='Specify whether to train or test model | Options: ["train", "test", "traintest"]', default='traintest')
parser.add_argument('--netG', help='Define Generator Architecture | Options: ["unet", "ResNet"]', default='unet')
parser.add_argument('--netD', help='Define Discriminator Architecture | Options: ["basic", "wgan", "cycle"]', default='basic')

parser.add_argument('--lambda_mse', help='MSELoss weight', type=float, default=1)
parser.add_argument('--lambda_adv', help='Adversarial Loss weight', type=float, default=0.01),

parser.add_argument('--lr', help='Learning Rate', type=float, default=0.0001)
parser.add_argument('--n_epochs', help='Number of epochs to train', type=int, default=4)
parser.add_argument('--trainG_freq', help='How often (epochs) to train generator', type=int, default=1)
parser.add_argument('--trainD_freq', help='How often (epochs) to train discriminator', type=int, default=1)
parser.add_argument('--print_freq', help='How many iters til stats print', type=int, default=20)
parser.add_argument('--plot_freq', help='How often (epochs) to display plots', type=int, default=1)
parser.add_argument('--save_freq', help='How often (epochs) to save checkpoint/results', type=int, default=1)
args = parser.parse_args()

if args.agave:
    DATA_DIR = '/scratch/jdrego/data/' + args.dataset + '/'
else:
    DATA_DIR = '/home/jdrego/PycharmProjects/DATASETS/' + args.dataset + '/'
# Specify Checkpoint and Result Directories
CHECKPOINT_DIR = './checkpoints/'+args.method+'/'+args.dataset+'/pretrain_adv'+str(int(args.lambda_adv*100))+'/'
RESULTS_DIR = './results/'+args.method+'/'+args.dataset+'/pretrain_adv'+str(int(args.lambda_adv*100))+'/'
# Checks if DIR exists, if not create DIR
utils.check_dir([CHECKPOINT_DIR, RESULTS_DIR])

# Define Generator Architecture | Options: ['unet', 'ResNet']
net_G = args.netG #net_G = 'unet'
# Discriminator Network Options: ['basic', 'cycle']
net_D = args.netD #net_D = 'basic'
#GPUtil.showUtilization()
# Assign GPU as device if available, else CPU
device = torch.device("cuda:"+args.train_gpu if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
print('Device:', device)
#GPUtil.showUtilization()
# ############################################
#     LOAD DATASET
# ############################################
if args.phase == 'train' or args.phase == 'debug_train':
    diff_loader, gt_loader, diff_val_loader, gt_val_loader = utils.load_data(DATA_DIR=DATA_DIR, batch_size=args.batch_size,
                                                                             num_workers=args.num_workers, phase=args.phase)
elif args.phase == 'test':
    diff_test_loader, gt_test_loader = utils.load_data(DATA_DIR=DATA_DIR, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, phase=args.phase)
elif args.phase == 'traintest':
    diff_loader, gt_loader, diff_val_loader, gt_val_loader, diff_test_loader, gt_test_loader = utils.load_data(DATA_DIR=DATA_DIR, batch_size=args.batch_size,
                                                                                                               num_workers=args.num_workers, phase=args.phase)
#GPUtil.showUtilization()
# ############################################
#     INITIALIZE/LOAD MODELS
# ############################################
model_G, model_D, opt_G, opt_D, scheduler_G, scheduler_D, losses, iter_losses, last_epoch = utils.load_models(netG=args.netG, netD=args.netD,
                                                                                                chkpoint=args.checkpoint, 
                                                                                                learning_rate=args.lr,
                                                                                                len_data=len(diff_loader.dataset),
                                                                                                batch_size=args.batch_size,
                                                                                                device=device, ngpu=args.ngpu,
                                                                                                CHECKPOINT_DIR=CHECKPOINT_DIR)
#GPUtil.showUtilization()
if args.debug:
    print('GENERATOR:'); print(model_G); print('DISCRIMINATOR:'); print(model_D)


if args.checkpoint > 0 or args.checkpoint == -1:
    utils.plot_losses(losses[:], name='epoch_loss', RESULTS_DIR=RESULTS_DIR, method=args.method)
    utils.plot_losses(iter_losses[0*len(diff_loader):], name='iter_loss', RESULTS_DIR=RESULTS_DIR, method=args.method)
# ############################################
#     TRAIN MODEL
# ############################################
if args.phase == 'train' or args.phase == 'traintest' or args.phase == 'debug_train':
    model_G = trainer.train(args.method, model_G, model_D, opt_G, opt_D, scheduler_G, scheduler_D,
                            diff_loader, gt_loader, diff_val_loader, gt_val_loader,
                            last_epoch, args.n_epochs, args.trainG_freq, args.trainD_freq, 
                            args.print_freq, args.plot_freq, args.save_freq,
                            args.lambda_mse, args.lambda_adv,losses, iter_losses,
                            device, args.ngpu, CHECKPOINT_DIR, RESULTS_DIR)
    last_epoch += args.n_epochs
#GPUtil.showUtilization()
# ############################################
#     TEST MODEL
# ############################################
if args.phase == 'test' or args.phase == 'traintest':
    tester.test(method=args.method, model_G=model_G, model_D=model_D, diff_loader=diff_test_loader, gt_loader=gt_test_loader,
                last_epoch=last_epoch, RESULTS_DIR=RESULTS_DIR, device=device)