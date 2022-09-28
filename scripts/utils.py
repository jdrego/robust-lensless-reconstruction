import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import os
import glob
#import cv2
#import scipy
#import skimage
import time
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision import utils as tvutils

from scripts import models

class NumpyDataset(Dataset):
	"""docstring for NumpyDataset"""
	def __init__(self, root_dir, transform=None):
		self.root_dir = root_dir
		self.transform = transform

def check_dir(PATHS):
	for path in PATHS:
		if not os.path.exists(path):
			os.makedirs(path)
			print(path, 'created')
		else:
			print(path, 'already exists')
    #print()

def np_loader(PATH):
	sample = np.load(PATH)
	return sample	

def normalize(I):
    I_norm = (I - np.amin(I))/(np.amax(I) - np.amin(I))
    return I_norm

def preplot(image):
    image = np.transpose(image, (1,2,0))
    image_color = np.zeros_like(image); 
    image_color[:,:,0] = image[:,:,2]; image_color[:,:,1]  = image[:,:,1]
    image_color[:,:,2] = image[:,:,0];
    out_image = np.flipud(np.clip(image_color, 0,1))
    return out_image[60:,62:-38,:]

def plot_loss(losses):
    fig, ax = plt.subplots(figsize=(12,8))
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
    plt.plot(losses.T[1], label='Generator_MSE+Adv', alpha=0.5)
    plt.plot(losses.T[2], label='Generators_MSE', alpha=0.5)
    plt.plot(losses.T[3], label='Generator_Adv', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
    plt.close() #plt.show()

def plot_losses(losses, name='figure', RESULTS_DIR='./', method='GAN'):
    if method == 'WGAN':
        rows, cols = (3, 2)
    else:
        rows, cols = (2, 2)
    fig, ax = plt.subplots(figsize=(20, 10))
    losses = np.asarray(losses)
    #print(losses.T)
    plt.subplot(rows, cols, 1)
    plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
    plt.title('Discriminator Loss')
    plt.ylabel('$L_{disc}$')
    plt.subplot(rows, cols, 2)
    plt.plot(losses.T[2], label='D_real', alpha=0.5)
    plt.plot(losses.T[3], label='D_fake', alpha=0.5)
    plt.title('Discriminator Output')
    plt.ylabel('$D(I)$')
    plt.legend()
    plt.subplot(rows, cols, 3)
    plt.plot(losses.T[1], label='Generator_MSE+Adv', alpha=0.5)
    plt.title('Generator Loss')
    plt.ylabel('$L_{gen}$')
    plt.subplot(rows, cols, 4)
    plt.plot(losses.T[4], label='Generators_MSE', alpha=0.5)
    plt.plot(losses.T[5], label='Generator_Adv', alpha=0.5)
    plt.title("Training Losses")
    plt.ylabel('$Loss$')
    plt.legend()
    if method == 'WGAN':
        plt.subplot(rows, cols, 5)
        plt.plot(losses.T[6], label='Wasserstein_Dist', alpha=0.5)
        plt.title("Wasserstein Distance")
        plt.ylabel('$Wass_Dist$')
        plt.subplot(rows, cols, 6)
        plt.plot(losses.T[7], label='Gradient_Penalty', alpha=0.5)
        plt.title("Gradient Penalty")
        plt.ylabel('$Grad_Penalty$')
    plt.savefig(RESULTS_DIR + name + '.png')
    plt.close()  # plt.show()

# ############################################
#     LOAD DATASET
# ############################################
def load_data(DATA_DIR, batch_size, num_workers, phase):
    # Directories for train and test data
    TRAIN_ROOT = DATA_DIR + 'train/'; VAL_ROOT = DATA_DIR + 'val/'; TEST_ROOT = DATA_DIR + 'test/'

    # convert data to torch.FloatTensor
    transform = transforms.Compose([transforms.ToTensor()])
    # load the training and test datasets
    if phase == 'train' or phase == 'traintest' or phase == 'debug_train':
        # Load Training dataset
        train_data = datasets.DatasetFolder(root=TRAIN_ROOT, loader=np_loader, transform=transform,
                                            extensions=('.npy', '.jpg'))
        val_data = datasets.DatasetFolder(root=VAL_ROOT, loader=np_loader, transform=transform,
                                            extensions=('.npy', '.jpg'))
        # Split Input and GT Training images
        # This only loads single batch to debug training
        if phase == 'debug_train':
            diff_data = torch.utils.data.Subset(train_data, list(range(batch_size)))
            gt_data = torch.utils.data.Subset(train_data,list(range(len(train_data) // 2, len(train_data)//2 + batch_size)))
            diff_val_data = torch.utils.data.Subset(train_data, list(range(batch_size)))
        # Loads entire dataset
        else:
            diff_data = torch.utils.data.Subset(train_data, list(range(len(train_data) // 2)))
            gt_data = torch.utils.data.Subset(train_data,list(range(len(train_data) // 2, len(train_data))))
            diff_val_data = torch.utils.data.Subset(val_data, list(range(len(val_data) // 2)))
            gt_val_data = torch.utils.data.Subset(val_data,list(range(len(val_data) // 2, len(val_data))))
        # Prepare Training data loaders
        diff_loader = DataLoader(diff_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        gt_loader = DataLoader(gt_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        diff_val_loader = DataLoader(diff_val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        gt_val_loader = DataLoader(gt_val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        
        print('Train Classes:', train_data.classes)
        print('# of TRAIN INPUT images:', len(diff_data)); print('# of TRAIN GT images:', len(gt_data))

    if phase == 'test' or phase == 'traintest':
        # Load Test dataset
        test_data = datasets.DatasetFolder(root=TEST_ROOT, loader=np_loader, transform=transform,
                                           extensions=('.npy','.jpg'))
        # Split Input and GT Test images
        diff_test_data = torch.utils.data.Subset(test_data, list(range(len(test_data) // 2)))
        gt_test_data = torch.utils.data.Subset(test_data, list(range(len(test_data) // 2, len(test_data))))
        # Prepare Test data loaders
        diff_test_loader = DataLoader(diff_test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        gt_test_loader = DataLoader(gt_test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        
        print('Test Classes:', test_data.classes)
        print('# of TEST INPUT images:', len(diff_test_data)); print('# of TEST GT images:', len(gt_test_data))

    if phase == 'train':
        return diff_loader, gt_loader, diff_val_loader, gt_val_loader
    elif phase == 'test':
        return diff_test_loader, gt_test_loader
    elif phase == 'traintest':
        return diff_loader, gt_loader, diff_val_loader, gt_val_loader, diff_test_loader, gt_test_loader

# ############################################
#     INITIALIZE/LOAD MODELS
# ############################################
def load_models(netG, netD, chkpoint, learning_rate, len_data, batch_size, device, ngpu, CHECKPOINT_DIR):
    # ====================
    # Training Options
    # ====================
    last_epoch = 0    #Initialize lastepoch
    base_lr = 1e-7
    max_lr = 0.0003
    #learning_rate = args.lr    #Define Learning Rate
    losses = []    # Initialize Epoch losses
    iter_losses = [] # Initialize iteration losses

    # ====================
    # Initialize Models
    # ====================
    # Initialize Generator and Discriminator Models
    model_G = models.define_G(netG)
    model_D = models.define_D(netD)
    model_G = model_G.to(device)
    model_D = model_D.to(device)
    
    # ONLY UNCOMMENT IF MODEL WAS SAVED AS DATA_PARALLEL
    #model_G = nn.DataParallel(model_G, list(range(ngpu)))
    #model_D = nn.DataParallel(model_D, list(range(ngpu)))
    # Initialize Optimizers
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=learning_rate)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=learning_rate)
    # Initialize LR Scheduler
    """
    scheduler_G = torch.optim.lr_scheduler.CyclicLR(optimizer_G, base_lr=base_lr, max_lr=0.0003,
                                                  step_size_up=(len_data//batch_size)*2, step_size_down=None, mode='triangular',
                                                  gamma=1.0, scale_fn=None, scale_mode='cycle',
                                                  cycle_momentum=False, base_momentum=0.8, max_momentum=0.9,
                                                  last_epoch=-1)
    """
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=4, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=4, gamma=0.5)
    # ====================
    # LOAD CHECKPOINT
    # ====================
    # Look for Checkpoints
    #print(CHECKPOINT_DIR)
    allcheckpoints = glob.glob(CHECKPOINT_DIR + '*.pt')
    #print('ALL CHECKPOINTS:', allcheckpoints)
    for checkpoint in allcheckpoints:
        checkpoint = os.path.basename(checkpoint)
        #print('Checkpoint:', int(checkpoint[4:-3]))
        last_epoch = np.maximum(last_epoch, int(checkpoint[:-3]))
    # Overide if desired
    #last_epoch = 0
    print('LASTEPOCH', last_epoch); print()
    # Load State Dictionaries if checkpoint is available
    if chkpoint == -2:
        last_checkpoint = torch.load(CHECKPOINT_DIR + '/pretrain/pretrained_140' + '.pt')
        model_G.load_state_dict(last_checkpoint['net'])
        last_epoch = 0
    elif chkpoint != 0:
        if chkpoint != -1: last_epoch = chkpoint
        last_checkpoint = torch.load(CHECKPOINT_DIR + str(last_epoch) + '.pt')
        model_G.load_state_dict(last_checkpoint['netG'])
        model_D.load_state_dict(last_checkpoint['netD'])
        optimizer_G.load_state_dict(last_checkpoint['optimizerG'])
        optimizer_D.load_state_dict(last_checkpoint['optimizerD'])
        #optimizer_G.param_groups[0]['lr'] = learning_rate
        scheduler_G.load_state_dict(last_checkpoint['schedulerG'])
        scheduler_D.load_state_dict(last_checkpoint['schedulerD'])
        #scheduler.base_lr = learning_rate; scheduler.base_lrs = [learning_rate] 
        #scheduler.max_lrs = [0.0003]
        losses = last_checkpoint['losses']
        iter_losses = last_checkpoint['iter_losses']
        print(optimizer_G.state_dict); print()
        print('SCHEDULER STATE_DICT')
        for var_name in scheduler_G.state_dict():
                print(var_name, "\t", scheduler_G.state_dict()[var_name])
        print()
        
    
    # ====================
    # Send to GPU
    # ====================
    #Handle multi-gpu if desired 
    if (device.type == 'cuda') and (ngpu > 1):                  # COMMENT OUT IF 
        model_G = nn.DataParallel(model_G, list(range(ngpu)))   # DATA WAS SAVED
        model_D = nn.DataParallel(model_D, list(range(ngpu)))   # AS DATA_PARALLEL

    return model_G, model_D, optimizer_G, optimizer_D, scheduler_G, scheduler_D, losses, iter_losses, last_epoch

def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE, device, LAMBDA=10):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()//BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 270, 480)
    alpha = alpha.to(device) if (device.type == 'cuda') else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device) if (device.type == 'cuda') else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
