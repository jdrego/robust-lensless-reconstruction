#!/usr/bin/env python
# coding: utf-8

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

import utils
import models

# ############################################
#     OPTIONS
# ############################################
parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='Prints debug info', action='store_true')
parser.add_argument('--agave', help='Specify whether running on AGAVE cluser', action='store_true')

parser.add_argument('--ngpu', help='Specify number of GPUs to use', type=int, default=1)
parser.add_argument('--batch_size', help='Specify number of samples in each batch iteration', type=int, default=20)
parser.add_argument('--num_workers', help='Specify num. of workers to load images', type=int, default=2)

parser.add_argument('--dataset', help='Specify Dataset | Options: ["Synthetic-COIL20", "diffusercam-berkeley", "diffcam-mini"]', default='diffusercam-berkeley')
parser.add_argument('--netG', help='Define Generator Architecture | Options: ["unet", "ResNet"]', default='unet')
parser.add_argument('--netD', help='Define Discriminator Architecture | Options: ["basic", "wgan", "cycle"]', default='basic')

parser.add_argument('--lambda_mse', help='MSELoss weight', type=float, default=1)
parser.add_argument('--lambda_adv', help='Adversarial Loss weight', type=float, default=0.6),

parser.add_argument('--lr', help='Learning Rate', type=float, default=0.000025)
parser.add_argument('--n_epochs', help='Number of epochs to train', type=int, default=4)
parser.add_argument('--trainG_freq', help='How often (epochs) to train generator', type=int, default=1)
parser.add_argument('--trainD_freq', help='How often (epochs) to train discriminator', type=int, default=1)
parser.add_argument('--print_freq', help='How many iters til stats print', type=int, default=20)
parser.add_argument('--plot_freq', help='How often (epochs) to display plots', type=int, default=4)
parser.add_argument('--save_freq', help='How often (epochs) to save checkpoint/results', type=int, default=1)
args = parser.parse_args()
# Determine whether to print debug info
#debug = args.debug #debug = True
# Training on AGAVE compute cluster
#agave = args.agave #agave = True
# Number of GPUs to use
#ngpu = args.ngpu #ngpu = 2
# how many samples per batch to load
#batch_size = args.batch_size #batch_size = 40
# Number of workers
#num_workers = args.num_workers #num_workers = 2

# Specify Dataset | Options: ['Synthetic-COIL20', 'diffusercam-berkeley', 'diffcam-mini']
#dataset = args.dataset #dataset = 'diffusercam-berkeley'; data_is_image = False
if args.agave:
	DATA_DIR = '/scratch/jdrego/data/' + args.dataset + '/'
else:
	DATA_DIR = '/home/jdrego/PycharmProjects/DATASETS/' + args.dataset + '/'
# Specify Checkpoint and Result Directories
CHECKPOINT_DIR = './checkpoints/GAN/' + args.dataset + '/adv' + str(int(args.lambda_adv*100)) + '/'
RESULTS_DIR = './results/GAN/' + args.dataset + '/adv' + str(int(args.lambda_adv*100)) + '/'
# Checks if DIR exists, if not create DIR
utils.check_dir([CHECKPOINT_DIR, RESULTS_DIR])     

# Define Generator Architecture | Options: ['unet', 'ResNet']
net_G = args.netG #net_G = 'unet'
# Discriminator Network Options: ['basic', 'cycle']
net_D = args.netD #net_D = 'basic'

# Assign GPU as device if available, else CPU
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
print('Device:', device)

# ############################################
#     LOAD DATASET
# ############################################

# Directories for train and test data
TRAIN_ROOT = DATA_DIR + 'train/'; TEST_ROOT = DATA_DIR + 'test/'

# convert data to torch.FloatTensor
transform = transforms.Compose([transforms.ToTensor()])
# load the training and test datasets
train_data = datasets.DatasetFolder(root=TRAIN_ROOT, loader=utils.np_loader, transform=transform,
                                   extensions=('.npy', '.jpg'))
test_data = datasets.DatasetFolder(root=TEST_ROOT, loader=utils.np_loader, transform=transform,
                                   extensions=('.npy','.jpg'))
print('Test Classes:', test_data.classes)
# Split Input, GT, and Test images
diff_data = torch.utils.data.Subset(train_data, list(range(len(train_data) // 2)))
gt_data = torch.utils.data.Subset(train_data,list(range(len(train_data) // 2, len(train_data))))
diff_test_data = torch.utils.data.Subset(test_data, list(range(len(test_data) // 2)))
gt_test_data = torch.utils.data.Subset(test_data, list(range(len(test_data) // 2, len(test_data))))
print('Train Classes:', train_data.classes)
print('# of TEST INPUT images:', len(diff_test_data)); print('# of TEST GT images:', len(gt_test_data))
print('# of TRAIN INPUT images:', len(diff_data)); print('# of TRAIN GT images:', len(gt_data))
# Prepare data loaders
diff_loader = DataLoader(diff_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
gt_loader = DataLoader(gt_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
diff_test_loader = DataLoader(diff_test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
gt_test_loader = DataLoader(gt_test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

# ############################################
#     SAMPLE DATASET IMAGES
# ############################################
"""    
# obtain one batch of training images
# dataiter = iter(train_loader)
dataiter = iter(diff_loader)
images, im_labels = dataiter.next()
gtiter = iter(gt_loader)
gt, gt_labels = gtiter.next()
#gt = gt.permute(0,2,3,1)
#images = images.permute(0, 2, 3, 1).numpy()
images, gt = images.numpy(), gt.numpy()
# get one image from the batch
img1 = np.squeeze(images[0])
img2 = np.squeeze(gt[0])
print(img1.shape)
fig = plt.figure(figsize = (10,10)) 
ax = fig.add_subplot(121)
ax.imshow(preplot(img1)/np.max(img1))
ax = fig.add_subplot(122)
ax.imshow(preplot(img2)/np.max(img2))

"""

# ############################################
#     LOSS EQUATIONS
# ############################################
# def real_mse_loss(D_out):
#     # how close is the produced output from being "real"?
#     return torch.mean((D_out-1)**2)

# def fake_mse_loss(D_out):
#     # how close is the produced output from being "fake"?
#     return torch.mean(D_out**2)

# def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):
#     # calculate reconstruction loss 
#     # as absolute value difference between the real and reconstructed images
#     reconstr_loss = torch.mean(torch.abs(real_im - reconstructed_im))
#     # return weighted loss
#     return lambda_weight*reconstr_loss   
# def loss_mse(reconstructed_im, real_im):
#     return torch.mean(torch.abs(real_im - reconstructed_im)**2)
# def loss_adv(D_out):
#     return (-torch.log(torch.mean(D_out)+1e-6))
def loss_disc_gt(D_out):
    return (-torch.log(torch.mean(D_out)+1e-6))
def loss_disc_est(D_out):
    return (-torch.log(1 - torch.mean(D_out)+1e-6))
def loss_MSE(D_out, label):
    criterion = nn.MSELoss()
    return criterion(D_out, label)

# ############################################
#     INITIALIZE/LOAD MODELS
# ############################################
# ====================
# Training Options
# ====================
last_epoch = 0    #Initialize lastepoch
base_lr = 1e-7
max_lr = 0.0003
learning_rate = args.lr    #Define Learning Rate
losses = []    # Initialize Epoch losses
iter_losses = [] # Initialize iteration losses

# ====================
# Initialize Models
# ====================
# Initialize Generator and Discriminator Models
model_G = models.define_G(net_G)
model_D = models.define_D(net_D)
model_G = model_G.to(device)
model_D = model_D.to(device)
# specify loss function
#criterion = nn.MSELoss()
# Initialize Optimizers
optimizer_G = torch.optim.Adam(model_G.parameters(), lr=learning_rate)
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=learning_rate)
# Initialize LR Scheduler
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer_G, base_lr=learning_rate, max_lr=0.0003,
#                                              step_size_up=(len(diff_data)//args.batch_size)*2, step_size_down=None, mode='triangular',
#                                              gamma=1.0, scale_fn=None, scale_mode='cycle',
#                                              cycle_momentum=False, base_momentum=0.8, max_momentum=0.9,
#                                              last_epoch=-1)
# ====================
# LOAD CHECKPOINT
# ====================
# Look for Checkpoints
allcheckpoints = glob.glob(CHECKPOINT_DIR + '*.pt')
#print('ALL CHECKPOINTS:', allcheckpoints)
for checkpoint in allcheckpoints:
    checkpoint = os.path.basename(checkpoint)
    #print('Checkpoint:', int(checkpoint[4:-3]))
    last_epoch = np.maximum(last_epoch, int(checkpoint[:-3]))
# Overide if desired
last_epoch = 0
print('LASTEPOCH', last_epoch); print()
# Load State Dictionaries if checkpoint is available
if last_epoch != 0:
    last_checkpoint = torch.load(CHECKPOINT_DIR + str(last_epoch) + '.pt')
    model_G.load_state_dict(last_checkpoint['netG'])
    model_D.load_state_dict(last_checkpoint['netD'])
    optimizer_G.load_state_dict(last_checkpoint['optimizerG'])
    optimizer_D.load_state_dict(last_checkpoint['optimizerD'])
    optimizer_G.param_groups[0]['lr'] = learning_rate
#    scheduler.load_state_dict(last_checkpoint['schedulerG'])
#    scheduler.base_lr = learning_rate; scheduler.base_lrs = [learning_rate] 
#    scheduler.max_lrs = [0.0003]
    losses = last_checkpoint['losses']
    iter_losses = last_checkpoint['iter_losses']
    print(optimizer_G.state_dict); print()
    print('SCHEDULER STATE_DICT')
#    for var_name in scheduler.state_dict():
#            print(var_name, "\t", scheduler.state_dict()[var_name])
    print()
# ====================
# Send to GPU
# ====================
#Handle multi-gpu if desired
if (device.type == 'cuda') and (args.ngpu > 1):
    model_G = nn.DataParallel(model_G, list(range(args.ngpu)))
    model_D = nn.DataParallel(model_D, list(range(args.ngpu)))

if args.debug:
    print('GENERATOR:'); print(model_G); print('DISCRIMINATOR:'); print(model_D)


if last_epoch != 0:
	utils.plot_losses(losses[:], 'epoch_loss', RESULTS_DIR)
	utils.plot_losses(iter_losses[0*len(diff_loader):], 'iter_loss', RESULTS_DIR)

# ############################################
#     TRAIN MODEL
# ############################################
model_G.train(); model_D.train()
# ===================
# Training Options
# ===================
# n_epochs = args.n_epochs	# number of epochs to train the model
# trainG_freq = 1				# How often (epochs) to train generator
# trainD_freq = 1				# How often (epochs) to train discriminator
# print_every = 20    		# How many iters til stats print
# disp_freq = n_epochs		# How often (epochs) to display plots
# save_freq = 1				# How often (epochs) to save checkpoint/results
n_epochs = args.n_epochs		# number of epochs to train the model
trainG_freq = args.trainG_freq	# How often (epochs) to train generator
trainD_freq = args.trainD_freq	# How often (epochs) to train discriminator
print_every = args.print_freq   # How many iters til stats print
disp_freq = args.plot_freq		# How often (epochs) to display plots
save_freq = args.save_freq		# How often (epochs) to save checkpoint/results
lambda_mse = args.lambda_mse; lambda_adv = args.lambda_adv

start_epoch = last_epoch+1; end_epoch = last_epoch + n_epochs
train_start = time.time()
for epoch in range(start_epoch, end_epoch+1):
    ep_start = time.time()
    iter_start = time.time()

    # monitor training loss
    train_loss, train_loss_G, train_loss_D, train_D_real, train_D_fake = 0.0, 0.0, 0.0, 0.0, 0.0
    adv_loss, mse_loss, = 0.0, 0.0
    #optimizer_G.param_groups[0]['lr'] = optimizer_G.param_groups[0]['lr'] * 0.90
    #optimizer_D.param_groups[0]['lr'] = optimizer_D.param_groups[0]['lr'] * 0.7
    print('START EPOCH:', epoch, 'LEARNING RATE:', optimizer_G.param_groups[0]['lr'])

    count_iter = 0
    for data, targ in zip(diff_loader, gt_loader):
        # Load input images and ground-truth targets
        images = data[0].to(device)
        target = targ[0].to(device)
        
        
        # ======================
        # TRAIN DISCRIMINATOR
        # ======================
        # Train with real images
        optimizer_D.zero_grad()
        # 1. Compute the real loss for D
        out = model_D(target)
        D_real_loss = loss_disc_gt(out)
        D_real = out.mean().item()
        # Train with fake images
        # 2. Reconstruct Lensless Images
        fake_imgs = model_G(images)
        # 3. Compute the fake loss for D
        out = model_D(fake_imgs)
        D_fake_loss = loss_disc_est(out)
        D_fake = out.mean().item()
        # 4. Compute the total loss and perform backprop
        loss_D = D_real_loss + D_fake_loss
        loss_D.backward()
        optimizer_D.step()

    #    iter_lossD = loss_D.item()*images.size(0)
        iter_lossD = loss_D.item()
        train_loss_D += iter_lossD 
        train_D_real += D_real
        train_D_fake += D_fake
        
        # ======================
        # TRAIN GENERATOR
        # ======================
        # clear the gradients of all optimized variables
        optimizer_G.zero_grad()
        ## forward pass: compute predicted outputs by passing *noisy* images to the model
        outputs = model_G(images)
        # calculate the loss
        G_mse_loss = lambda_mse * loss_MSE(outputs, target)
        #G_mse_loss.backward()
        # the "target" is still the original, not-noisy images
        pred_fake = model_D(outputs)
        G_adv_loss = lambda_adv * loss_disc_gt(pred_fake)
        
        #G_adv_loss.backward()
        loss_G = G_mse_loss + G_adv_loss
        
        if epoch % trainG_freq == 0:
            # backward pass: compute gradient of the loss with respect to model parameters
            loss_G.backward()
            # perform a single optimization step (parameter update)
            optimizer_G.step()

        # update running training loss
    #    iter_lossG = loss_G.item()*images.size(0)
    #    iter_loss_mse = G_mse_loss.item()*images.size(0)
    #    iter_loss_adv = G_adv_loss.item()*images.size(0)
        iter_lossG = loss_G.item()
        iter_loss_mse = G_mse_loss.item()
        iter_loss_adv = G_adv_loss.item()
        train_loss_G += iter_lossG
        mse_loss += iter_loss_mse
        adv_loss += iter_loss_adv
        
        count_iter += 1
        if count_iter % print_every == 0 or count_iter < 5:
            iter_end = time.time() - iter_start
            iter_losses.append((iter_lossD, iter_lossG, D_real, D_fake, G_mse_loss.item(), G_adv_loss.item()))
            print('[{:4d}/{:4d}][{:5d}/{:5d}] | LR: | loss_D: {:6.3f} | loss_G: {:6.3f} | D_real: {:2.3f} | D_fake: {:2.3f} | G_MSE_loss: {:6.3f} | G_adv_Loss: {:6.3f} | Time: {}h:{}m:{:2.3f}s'.format(
                epoch, end_epoch, count_iter * args.batch_size, len(train_data)//2, #float(scheduler.get_lr()[0]),
                iter_lossD, iter_lossG, D_real, D_fake, G_mse_loss.item(), G_adv_loss.item(), 
                int(iter_end//3600), int((iter_end//60) % 60), iter_end % 60))
            iter_start = time.time()
        #scheduler.step()
             
    ep_end = time.time() - ep_start
    
    # Print the log info
    if epoch % 1 == 0:
        # append real and fake discriminator losses and the generator loss
        train_loss_D, train_loss_G = train_loss_D / len(diff_loader), train_loss_G / len(diff_loader)
        G_mse_loss, G_adv_loss =  mse_loss / len(diff_loader), adv_loss / len(diff_loader)
        train_D_real, train_D_fake = train_D_real / len(diff_loader), train_D_fake / len(diff_loader)
        losses.append((train_loss_D, train_loss_G, train_D_real, train_D_fake, G_mse_loss, G_adv_loss))
        print('Epoch [{:4d}/{:4d}] | loss_D: {:6.4f} | loss_G: {:6.5f} | D_real: {:2.3f} | D_fake: {:2.3f} | G_MSE_loss: {:6.5f} | G_Adv_loss: {:6.5f} | Time: {}h:{}m:{:2.3f}s'.format(
            epoch, end_epoch, 
            train_loss_D, train_loss_G, train_D_real, train_D_fake, G_mse_loss, G_adv_loss, 
            int(ep_end//3600), int((ep_end//60) % 60), ep_end % 60))
        print()
        """
    # ==============================
    #  DISPLAY INTERMEDIATE RESULTS 
    # ==============================
	
	if epoch % disp_freq == 0 or epoch == start_epoch or epoch == end_epoch:
        print('')
        # prep images for display
        #noisy_imgs = noisy_imgs.numpy()
        images_np = images.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        # output is resized into a batch of iages
        #output = output.view(batch_size, 1, 28, 28)
        # use detach when it's an output that requires_grad
        output_np = outputs.detach().cpu().numpy()
        
        # plot the first ten input images and then reconstructed images
        fig, axes = plt.subplots(nrows=3, ncols=min(10,batch_size), sharex=True, sharey=True, figsize=(50,20))
        
        # input images on top row, reconstructions on bottom
        for noisy_imgs, row in zip([images_np, output_np, target_np], axes):
            for img, ax in zip(noisy_imgs, row):
                ax.imshow(preplot(normalize(np.squeeze(img))))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        plt.show()
        plot_losses(losses[-20:])
        plot_losses(iter_losses)
		"""      
    # ==============================
    #  SAVE RESULTS & CHECKPOINT 
    # ==============================                
    if epoch % save_freq == 0:
        RESULTS_TRAIN_PATH = RESULTS_DIR + 'train/'
        utils.check_dir([RESULTS_TRAIN_PATH, RESULTS_TRAIN_PATH+str(epoch),
        				 RESULTS_TRAIN_PATH+'input/', RESULTS_TRAIN_PATH+'gt/'])
        
        output_out = outputs.detach().cpu().numpy()
        images_out = images.detach().cpu().numpy()
        target_out = target.detach().cpu().numpy()
        
        #print(output_out.shape)
        for i in range (args.batch_size):
            reconstruction = Image.fromarray((utils.normalize(utils.preplot(np.squeeze(output_out[i,:,:,:]))) * 255).astype('uint8'))
            imgs = Image.fromarray((utils.normalize(utils.preplot(np.squeeze(images_out[i,:,:,:]))) * 255).astype('uint8'))
            gt = Image.fromarray((utils.normalize(utils.preplot(np.squeeze(target_out[i,:,:,:]))) * 255).astype('uint8'))
            
            reconstruction.save(RESULTS_TRAIN_PATH+str(epoch)+'/'+str(i)+'.png')
            imgs.save(RESULTS_TRAIN_PATH+'input/'+str(i)+'.png')
            gt.save(RESULTS_TRAIN_PATH+'gt/'+str(i)+'.png')
            
        CHECKPOINT_PATH = CHECKPOINT_DIR + str(epoch) + '.pt'
        if args.ngpu > 1:
        	netG_state = model_G.module.state_dict()
        	netD_state = model_D.module.state_dict()
        else:
        	netG_state = model_G.state_dict()
        	netD_state = model_D.state_dict()
        checkpoint_state = {'netG': model_G.state_dict(),
							'netD': model_D.state_dict(),
                            'optimizerG': optimizer_G.state_dict(),
                            'optimizerD': optimizer_D.state_dict(),
                            #'schedulerG': scheduler.state_dict(),
                            'losses': losses,
                            'iter_losses': iter_losses}
        torch.save(checkpoint_state, CHECKPOINT_PATH)
    last_epoch = epoch
train_end = time.time() - train_start
print('Finished Training {} epochs [{}-{}] | Time taken: {}h:{}m:{:2.3f}s'.format(
    n_epochs, start_epoch, last_epoch, int(train_end//3600), int((train_end//60) % 60), train_end % 60))
# Clear gpu cache
if (device.type == 'cuda'):
	torch.cuda.empty_cache()

# Plot losses
utils.plot_losses(losses[:], 'epoch loss', RESULTS_DIR)
utils.plot_losses(iter_losses, 'iter loss', RESULTS_DIR)

# ############################################
#     TEST MODEL
# ############################################
device = torch.device("cuda:0")
model = model_G.to(device)
model.eval()

# obtain one batch of test images
dataiter = iter(diff_test_loader)
images_test = dataiter.next()[0]
dataiter = iter(gt_test_loader)
target_test, _ = dataiter.next()
# get sample outputs
output_test = model_G(images_test.to(device))

images_np = images_test.numpy()
# use detach when it's an output that requires_grad
output_np = output_test.detach().cpu().numpy()
gt_np = target_test.numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(50,10))

# input images on top row, reconstructions on bottom
for noisy_imgs, row in zip([images_np, output_np, gt_np], axes):
    for img, ax in zip(noisy_imgs, row):
        ax.imshow(utils.normalize(utils.preplot(np.squeeze(img))), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
RESULTS_TEST_PATH = RESULTS_DIR + 'test/'
utils.check_dir([RESULTS_TEST_PATH, RESULTS_TEST_PATH+str(last_epoch), RESULTS_TEST_PATH+'input/', RESULTS_TEST_PATH+'gt/'])

for i in range (args.batch_size):
    input_img = Image.fromarray((utils.normalize(utils.preplot(np.squeeze(images_np[i,:,:,:]))) * 255).astype('uint8'))
    reconstruction = Image.fromarray((utils.normalize(utils.preplot(np.squeeze(output_np[i,:,:,:]))) * 255).astype('uint8'))
    gt_img = Image.fromarray((utils.normalize(utils.preplot(np.squeeze(gt_np[i,:,:,:]))) * 255).astype('uint8'))
    
    input_img.save(RESULTS_TEST_PATH + 'input/' + str(i) + '.png')
    reconstruction.save(RESULTS_TEST_PATH + str(last_epoch) + '/' +str(i)+'.png')
    gt_img.save(RESULTS_TEST_PATH + 'gt/' + str(i) + '.png')
print('Test images written to', RESULTS_TEST_PATH)
# Clear gpu cache
if (device.type == 'cuda'):
	torch.cuda.empty_cache()






