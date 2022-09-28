#!/usr/bin/env python
# coding: utf-8
import GPUtil
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
import tester

# ############################################
#     TRAIN MODEL
# ############################################
# ===================
# Training Losses
# ===================
def loss_disc_gt(D_out):
    return (-torch.log(torch.mean(D_out)+1e-6))
def loss_disc_est(D_out):
    return (-torch.log(1 - torch.mean(D_out)+1e-6))
def loss_MSE(D_out, label):
    criterion = nn.MSELoss()
    return criterion(D_out, label)
    
# ===================
# Trainer
# ===================
def train(method, model_G, model_D, optimizer_G, optimizer_D, scheduler_G, scheduler_D,
          diff_loader, gt_loader, diff_val_loader, gt_val_loader,
          last_epoch, n_epochs, trainG_freq, trainD_freq,
          print_freq, plot_freq, save_freq, 
          lambda_mse, lambda_adv, losses, iter_losses,
          device, ngpu, CHECKPOINT_DIR, RESULTS_DIR):
    RESULTS_TRAIN_PATH = RESULTS_DIR + 'train/'
    utils.check_dir([RESULTS_TRAIN_PATH, RESULTS_TRAIN_PATH+'input/', RESULTS_TRAIN_PATH+'gt/'])
    model_G.train(); model_D.train()
    # ===================
    # Training Definitions
    # ===================
    # n_epochs = args.n_epochs    # number of epochs to train the model
    # trainG_freq = 1                # How often (epochs) to train generator
    # trainD_freq = 1                # How often (epochs) to train discriminator
    # print_every = 20            # How many iters til stats print
    # disp_freq = n_epochs        # How often (epochs) to display plots
    # save_freq = 1                # How often (epochs) to save checkpoint/results
    # n_epochs = args.n_epochs        # number of epochs to train the model
    # trainG_freq = args.trainG_freq    # How often (epochs) to train generator
    # trainD_freq = args.trainD_freq    # How often (epochs) to train discriminator
    # print_every = args.print_freq   # How many iters til stats print
    # disp_freq = args.plot_freq        # How often (epochs) to display plots
    # save_freq = args.save_freq        # How often (epochs) to save checkpoint/results
    # lambda_mse = args.lambda_mse; lambda_adv = args.lambda_adv
    #GPUtil.showUtilization()
    if method == 'WGAN':
        one = torch.tensor(1, dtype=torch.float) #one = torch.FloatTensor([1])
        mone = one * -1
        one = one.to(device)
        mone = mone.to(device)
    if last_epoch == 0:
        iter_losses = [[0,0,0,0,0,0,0,0]]
    min_val_loss = losses[-1][-1] if (last_epoch != 0) else np.inf
    start_epoch = last_epoch+1; end_epoch = last_epoch + n_epochs
    train_start = time.time()
    for epoch in range(start_epoch, end_epoch+1):
        ep_start = time.time()
        iter_start = time.time()

        # monitor training loss
        train_loss_G, train_loss_D, train_D_real, train_D_fake = 0.0, 0.0, 0.0, 0.0
        adv_loss, mse_loss, = 0.0, 0.0
        if method == 'WGAN':
            train_wass_D, train_gp = 0.0, 0.0
        #optimizer_G.param_groups[0]['lr'] = optimizer_G.param_groups[0]['lr'] * 0.90
        #optimizer_D.param_groups[0]['lr'] = optimizer_D.param_groups[0]['lr'] * 0.7
        print('START EPOCH:', epoch, 'LEARNING RATE:', optimizer_G.param_groups[0]['lr'])
        #GPUtil.showUtilization()
        count_iter = 0
        for data, targ in zip(diff_loader, gt_loader):
            #optimizer_D.zero_grad()
            count_iter += 1
            # Load input images and ground-truth targets
            images = data[0].to(device)
            #GPUtil.showUtilization()
            target = targ[0].to(device)
            #GPUtil.showUtilization()
            
            # ======================
            # TRAIN DISCRIMINATOR
            # ======================
            if count_iter % trainD_freq == 0:
                for p in model_D.parameters():
                    p.requires_grad = True
                # Train with real images
                optimizer_D.zero_grad()
                # 1. Compute the real loss for D
                out_real = model_D(target)
                if method == 'WGAN':
                    D_real = out_real.mean()
                    D_real.backward(mone)
                else:
                    D_real = loss_disc_gt(out_real)
                
                # Train with fake images
                # 2. Reconstruct Lensless Images
                fake_imgs = model_G(images)
                # 3. Compute the fake loss for D
                out_fake = model_D(fake_imgs)
                if method == 'WGAN':
                    D_fake = out_fake.mean()
                    D_fake.backward(one)
                else:
                    D_fake = loss_disc_est(out_fake)
                
                if method == 'WGAN':
                    gradient_penalty = utils.calc_gradient_penalty(model_D, target.data, fake_imgs.data, diff_loader.batch_size, device)
                    gradient_penalty.backward()
                    
                    loss_D = D_fake - D_real + gradient_penalty
                    Wasserstein_D = D_fake - D_real
                    
                    iter_wass_D = Wasserstein_D.item()
                    iter_gp = gradient_penalty.item()
                    train_wass_D += iter_wass_D
                    train_gp += iter_gp
                else:
                    # 4. Compute the total loss and perform backprop
                    loss_D = D_real + D_fake
                    
                optimizer_D.step()

                iter_lossD = loss_D.item()
                D_real_loss = D_real.item()
                D_fake_loss = D_fake.item()
                
            else:
                iter_lossD, D_real_loss, D_fake_loss = iter_losses[-1][0], iter_losses[-1][2], iter_losses[-1][3] 
            
            train_loss_D += iter_lossD 
            train_D_real += D_real_loss
            train_D_fake += D_fake_loss
            # ======================
            # TRAIN GENERATOR
            # ======================
            if count_iter % trainG_freq == 0:
                for p in model_D.parameters():
                    p.requires_grad = False
                # clear the gradients of all optimized variables
                optimizer_G.zero_grad()
                ## forward pass: compute predicted outputs by passing *noisy* images to the model
                outputs = model_G(images)
                # calculate the loss
                G_mse_loss = lambda_mse * loss_MSE(outputs, target)
                G_mse_loss.backward(retain_graph=True)
                # the "target" is still the original, not-noisy images
                pred_fake = model_D(outputs)
                if method == 'WGAN':
                    G_adv_loss = lambda_adv*pred_fake.mean()
                    G_adv_loss.backward(mone)
                    G_adv_loss = -G_adv_loss
                else:
                    G_adv_loss = loss_disc_gt(pred_fake)
                    G_adv_loss.backward()
                
                #G_adv_loss.backward()
                loss_G = G_mse_loss + G_adv_loss
            
            #if epoch % trainG_freq == 0:
                # backward pass: compute gradient of the loss with respect to model parameters
                #loss_G.backward()
                # perform a single optimization step (parameter update)
                optimizer_G.step()

                iter_lossG = loss_G.item()
                iter_loss_mse = G_mse_loss.item()
                iter_loss_adv = G_adv_loss.item()
            else:
                iter_lossG, iter_loss_mse, iter_loss_adv = iter_losses[-1][1], iter_losses[-1][4], iter_losses[-1][5] 

            # update running training loss
            train_loss_G += iter_lossG
            mse_loss += iter_loss_mse
            adv_loss += iter_loss_adv
            
            
            if count_iter % print_freq == 0 or count_iter < 5:
                iter_end = time.time() - iter_start
                if method == 'WGAN':
                    iter_losses.append((iter_lossD, iter_lossG, D_real_loss, D_fake_loss, iter_loss_mse, iter_loss_adv, iter_wass_D, iter_gp))
                else:
                    iter_losses.append((iter_lossD, iter_lossG, D_real_loss, D_fake_loss, iter_loss_mse, iter_loss_adv))
                print('[{:4d}/{:4d}][{:5d}/{:5d}] | LR: {:1.3e} | loss_D: {:6.3f} | loss_G: {:6.3f} | D_real: {:2.3f} | D_fake: {:2.3f} | G_MSE_loss: {:6.3f} | G_adv_Loss: {:6.3f} | Time: {}h:{}m:{:2.3f}s'.format(
                    epoch, end_epoch, count_iter * diff_loader.batch_size, len(diff_loader.dataset), float(scheduler_G.get_lr()[0]),
                    iter_lossD, iter_lossG, D_real, D_fake, iter_loss_mse, iter_loss_adv, 
                    int(iter_end//3600), int((iter_end//60) % 60), iter_end % 60))
                iter_start = time.time()
                
        # INDENT TO ITERATION IF CyclicLR Scheduled
        #scheduler_G.step(); scheduler_D.step()
            
        val_loss = tester.test(method=method, model_G=model_G, model_D=model_D, 
                               diff_loader=diff_val_loader, gt_loader=gt_val_loader,
                               last_epoch=epoch, RESULTS_DIR=RESULTS_DIR, device=device)
        if val_loss <= min_val_loss:
            print('Validation loss reduced on EPOCH:', epoch)
            min_val_loss = val_loss
        ep_end = time.time() - ep_start
        
        # Print the log info
        if epoch % 1 == 0:
            # append real and fake discriminator losses and the generator loss
            train_loss_D, train_loss_G = train_loss_D / len(diff_loader), train_loss_G / len(diff_loader)
            G_mse_loss, G_adv_loss =  mse_loss / len(diff_loader), adv_loss / len(diff_loader)
            train_D_real, train_D_fake = train_D_real / len(diff_loader), train_D_fake / len(diff_loader)
            if method == 'WGAN':
                train_wass_D, train_gp = train_wass_D / len(diff_loader), train_gp / len(diff_loader)
                losses.append((train_loss_D, train_loss_G, train_D_real, train_D_fake, G_mse_loss, G_adv_loss, train_wass_D, train_gp, val_loss))
            else:
                losses.append((train_loss_D, train_loss_G, train_D_real, train_D_fake, G_mse_loss, G_adv_loss, val_loss))
            print('Epoch [{:4d}/{:4d}] | loss_D: {:6.4f} | loss_G: {:6.5f} | D_real: {:2.3f} | D_fake: {:2.3f} | G_MSE_loss: {:6.5f} | G_Adv_loss: {:6.5f} | Val_loss: {:6.5f} | Time: {}h:{}m:{:2.3f}s'.format(
                epoch, end_epoch, 
                train_loss_D, train_loss_G, train_D_real, train_D_fake, G_mse_loss, G_adv_loss, val_loss,
                int(ep_end//3600), int((ep_end//60) % 60), ep_end % 60))
            print()

        # ==============================
        #  DISPLAY INTERMEDIATE RESULTS 
        # ==============================
        if epoch % plot_freq == 0:# or epoch == start_epoch or epoch == end_epoch:
            print('')
            # prep images for display
            images_np = images.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            output_np = outputs.detach().cpu().numpy()
            #output_np = fake_imgs.detach().cpu().numpy()
            
            # plot the first ten input images and then reconstructed images
            fig, axes = plt.subplots(nrows=3, ncols=min(10,diff_loader.batch_size), sharex=True, sharey=True, figsize=(50,20))
            # input images on top row, reconstructions on bottom
            for noisy_imgs, row in zip([images_np, output_np, target_np], axes):
                for img, ax in zip(noisy_imgs, row):
                    ax.imshow(utils.preplot(utils.normalize(np.squeeze(img))))
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
            plt.savefig(RESULTS_TRAIN_PATH + str(epoch) + '.png')
            #plt.show()
            plt.close()
            utils.plot_losses(losses[:], name='epoch_loss', RESULTS_DIR=RESULTS_DIR, method=method)
            utils.plot_losses(iter_losses[:], name='iter_loss', RESULTS_DIR=RESULTS_DIR, method=method)

        # ==============================
        #  SAVE RESULTS & CHECKPOINT 
        # ==============================                
        if epoch % save_freq == 0:
            utils.check_dir([RESULTS_TRAIN_PATH+str(epoch)])
            
            output_out = outputs.detach().cpu().numpy()
            images_out = images.detach().cpu().numpy()
            target_out = target.detach().cpu().numpy()
            
            #print(output_out.shape)
            for i in range (diff_loader.batch_size):
                reconstruction = Image.fromarray((utils.normalize(utils.preplot(np.squeeze(output_out[i,:,:,:]))) * 255).astype('uint8'))
                imgs = Image.fromarray((utils.normalize(utils.preplot(np.squeeze(images_out[i,:,:,:]))) * 255).astype('uint8'))
                gt = Image.fromarray((utils.normalize(utils.preplot(np.squeeze(target_out[i,:,:,:]))) * 255).astype('uint8'))
                
                reconstruction.save(RESULTS_TRAIN_PATH+str(epoch)+'/'+str(i)+'.png')
                imgs.save(RESULTS_TRAIN_PATH+'input/'+str(i)+'.png')
                gt.save(RESULTS_TRAIN_PATH+'gt/'+str(i)+'.png')
                    
            CHECKPOINT_PATH = CHECKPOINT_DIR + str(epoch) + '.pt'
            if ngpu > 1:
                netG_state = model_G.module.state_dict()
                netD_state = model_D.module.state_dict()
            else:
                netG_state = model_G.state_dict()
                netD_state = model_D.state_dict()
            checkpoint_state = {'netG': netG_state,
                                'netD': netD_state,
                                'optimizerG': optimizer_G.state_dict(),
                                'optimizerD': optimizer_D.state_dict(),
                                'schedulerG': scheduler_G.state_dict(),
                                'schedulerD': scheduler_D.state_dict(),
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

    return model_G

def pretrain_gen(method, model_G, optimizer_G, scheduler_G,
          diff_loader, gt_loader, diff_val_loader, gt_val_loader,
          last_epoch, n_epochs, trainG_freq, trainD_freq,
          print_freq, plot_freq, save_freq, 
          lambda_mse, lambda_adv, losses, iter_losses,
          device, ngpu, CHECKPOINT_DIR, RESULTS_DIR):
    RESULTS_TRAIN_PATH = RESULTS_DIR + 'train/'
    utils.check_dir([RESULTS_TRAIN_PATH, RESULTS_TRAIN_PATH+'input/', RESULTS_TRAIN_PATH+'gt/'])
    model_G.train(); model_D.train()
    # ===================
    # Training Definitions
    # ===================
    # n_epochs = args.n_epochs    # number of epochs to train the model
    # trainG_freq = 1                # How often (epochs) to train generator
    # trainD_freq = 1                # How often (epochs) to train discriminator
    # print_every = 20            # How many iters til stats print
    # disp_freq = n_epochs        # How often (epochs) to display plots
    # save_freq = 1                # How often (epochs) to save checkpoint/results
    # n_epochs = args.n_epochs        # number of epochs to train the model
    # trainG_freq = args.trainG_freq    # How often (epochs) to train generator
    # trainD_freq = args.trainD_freq    # How often (epochs) to train discriminator
    # print_every = args.print_freq   # How many iters til stats print
    # disp_freq = args.plot_freq        # How often (epochs) to display plots
    # save_freq = args.save_freq        # How often (epochs) to save checkpoint/results
    # lambda_mse = args.lambda_mse; lambda_adv = args.lambda_adv
    #GPUtil.showUtilization()
    """
    if method == 'WGAN':
        one = torch.tensor(1, dtype=torch.float) #one = torch.FloatTensor([1])
        mone = one * -1
        one = one.to(device)
        mone = mone.to(device)
        """
    if last_epoch == 0:
        iter_losses = [[0,0,0,0,0,0,0,0]]
    min_val_loss = losses[-1][-1] if (last_epoch != 0) else np.inf
    start_epoch = last_epoch+1; end_epoch = last_epoch + n_epochs
    train_start = time.time()
    for epoch in range(start_epoch, end_epoch+1):
        ep_start = time.time()
        iter_start = time.time()

        # monitor training loss
        train_loss_G, train_loss_D, train_D_real, train_D_fake = 0.0, 0.0, 0.0, 0.0
        adv_loss, mse_loss, = 0.0, 0.0
        
        #optimizer_G.param_groups[0]['lr'] = optimizer_G.param_groups[0]['lr'] * 0.90
        #optimizer_D.param_groups[0]['lr'] = optimizer_D.param_groups[0]['lr'] * 0.7
        print('START EPOCH:', epoch, 'LEARNING RATE:', optimizer_G.param_groups[0]['lr'])
        #GPUtil.showUtilization()
        count_iter = 0
        for data, targ in zip(diff_loader, gt_loader):
            #optimizer_D.zero_grad()
            count_iter += 1
            # Load input images and ground-truth targets
            images = data[0].to(device)
            #GPUtil.showUtilization()
            target = targ[0].to(device)
            #GPUtil.showUtilization()
            
            # ======================
            # TRAIN DISCRIMINATOR
            # ======================
            if count_iter % trainD_freq == 0:
                """
                for p in model_D.parameters():
                    p.requires_grad = True
                # Train with real images
                optimizer_D.zero_grad()
                # 1. Compute the real loss for D
                out_real = model_D(target)
                if method == 'WGAN':
                    D_real = out_real.mean()
                    D_real.backward(mone)
                else:
                    D_real = loss_disc_gt(out_real)
                
                # Train with fake images
                # 2. Reconstruct Lensless Images
                fake_imgs = model_G(images)
                # 3. Compute the fake loss for D
                out_fake = model_D(fake_imgs)
                if method == 'WGAN':
                    D_fake = out_fake.mean()
                    D_fake.backward(one)
                else:
                    D_fake = loss_disc_est(out_fake)
                
                if method == 'WGAN':
                    gradient_penalty = utils.calc_gradient_penalty(model_D, target.data, fake_imgs.data, diff_loader.batch_size, device)
                    gradient_penalty.backward()
                    
                    loss_D = D_fake - D_real + gradient_penalty
                    Wasserstein_D = D_fake - D_real
                    
                    iter_wass_D = Wasserstein_D.item()
                    iter_gp = gradient_penalty.item()
                    train_wass_D += iter_wass_D
                    train_gp += iter_gp
                else:
                    # 4. Compute the total loss and perform backprop
                    loss_D = D_real + D_fake
                    
                optimizer_D.step()

                iter_lossD = loss_D.item()
                D_real_loss = D_real.item()
                D_fake_loss = D_fake.item()
                
            else:
                iter_lossD, D_real_loss, D_fake_loss = iter_losses[-1][0], iter_losses[-1][2], iter_losses[-1][3] 
            
            train_loss_D += iter_lossD 
            train_D_real += D_real_loss
            train_D_fake += D_fake_loss
            """
            # ======================
            # TRAIN GENERATOR
            # ======================
            if count_iter % trainG_freq == 0:
                """
                for p in model_D.parameters():
                    p.requires_grad = False
                """
                # clear the gradients of all optimized variables
                optimizer_G.zero_grad()
                ## forward pass: compute predicted outputs by passing *noisy* images to the model
                outputs = model_G(images)
                # calculate the loss
                G_mse_loss = lambda_mse * loss_MSE(outputs, target)
                G_mse_loss.backward()
                # the "target" is still the original, not-noisy images
                """
                pred_fake = model_D(outputs)
                if method == 'WGAN':
                    G_adv_loss = pred_fake.mean()
                    G_adv_loss.backward(mone)
                    G_adv_loss = -G_adv_loss
                else:
                    G_adv_loss = loss_disc_gt(pred_fake)
                    G_adv_loss.backward()
                
                #G_adv_loss.backward()
                loss_G = lambda_mse*G_mse_loss + lambda_adv*G_adv_loss
                """
            #if epoch % trainG_freq == 0:
                # backward pass: compute gradient of the loss with respect to model parameters
                #loss_G.backward()
                # perform a single optimization step (parameter update)
                optimizer_G.step()

                iter_lossG = loss_G.item()
                iter_loss_mse = G_mse_loss.item()
                iter_loss_adv = G_adv_loss.item()
            else:
                iter_lossG, iter_loss_mse, iter_loss_adv = iter_losses[-1][1], iter_losses[-1][4], iter_losses[-1][5] 

            # update running training loss
            train_loss_G += iter_lossG
            mse_loss += iter_loss_mse
            adv_loss += iter_loss_adv
            
            
            if count_iter % print_freq == 0 or count_iter < 5:
                iter_end = time.time() - iter_start
                if method == 'WGAN':
                    iter_losses.append((iter_lossD, iter_lossG, D_real_loss, D_fake_loss, iter_loss_mse, iter_loss_adv, iter_wass_D, iter_gp))
                else:
                    iter_losses.append((iter_lossD, iter_lossG, D_real_loss, D_fake_loss, iter_loss_mse, iter_loss_adv))
                print('[{:4d}/{:4d}][{:5d}/{:5d}] | LR: {:1.3e} | loss_D: {:6.3f} | loss_G: {:6.3f} | D_real: {:2.3f} | D_fake: {:2.3f} | G_MSE_loss: {:6.3f} | G_adv_Loss: {:6.3f} | Time: {}h:{}m:{:2.3f}s'.format(
                    epoch, end_epoch, count_iter * diff_loader.batch_size, len(diff_loader.dataset), float(scheduler_G.get_lr()[0]),
                    iter_lossD, iter_lossG, D_real, D_fake, iter_loss_mse, iter_loss_adv, 
                    int(iter_end//3600), int((iter_end//60) % 60), iter_end % 60))
                iter_start = time.time()
                
        # INDENT TO ITERATION IF CyclicLR Scheduled
        #scheduler_G.step(); scheduler_D.step()
            
        val_loss = tester.test(method=method, model_G=model_G, model_D=model_D, 
                               diff_loader=diff_val_loader, gt_loader=gt_val_loader,
                               last_epoch=epoch, RESULTS_DIR=RESULTS_DIR, device=device)
        if val_loss <= min_val_loss:
            print('Validation loss reduced on EPOCH:', epoch)
            min_val_loss = val_loss
        ep_end = time.time() - ep_start
        
        # Print the log info
        if epoch % 1 == 0:
            # append real and fake discriminator losses and the generator loss
            train_loss_D, train_loss_G = train_loss_D / len(diff_loader), train_loss_G / len(diff_loader)
            G_mse_loss, G_adv_loss =  mse_loss / len(diff_loader), adv_loss / len(diff_loader)
            train_D_real, train_D_fake = train_D_real / len(diff_loader), train_D_fake / len(diff_loader)
            if method == 'WGAN':
                train_wass_D, train_gp = train_wass_D / len(diff_loader), train_gp / len(diff_loader)
                losses.append((train_loss_D, train_loss_G, train_D_real, train_D_fake, G_mse_loss, G_adv_loss, train_wass_D, train_gp, val_loss))
            else:
                losses.append((train_loss_D, train_loss_G, train_D_real, train_D_fake, G_mse_loss, G_adv_loss, val_loss))
            print('Epoch [{:4d}/{:4d}] | loss_D: {:6.4f} | loss_G: {:6.5f} | D_real: {:2.3f} | D_fake: {:2.3f} | G_MSE_loss: {:6.5f} | G_Adv_loss: {:6.5f} | Val_loss: {:6.5f} | Time: {}h:{}m:{:2.3f}s'.format(
                epoch, end_epoch, 
                train_loss_D, train_loss_G, train_D_real, train_D_fake, G_mse_loss, G_adv_loss, val_loss,
                int(ep_end//3600), int((ep_end//60) % 60), ep_end % 60))
            print()

        # ==============================
        #  DISPLAY INTERMEDIATE RESULTS 
        # ==============================
        if epoch % plot_freq == 0:# or epoch == start_epoch or epoch == end_epoch:
            print('')
            # prep images for display
            images_np = images.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            output_np = outputs.detach().cpu().numpy()
            #output_np = fake_imgs.detach().cpu().numpy()
            
            # plot the first ten input images and then reconstructed images
            fig, axes = plt.subplots(nrows=3, ncols=min(10,diff_loader.batch_size), sharex=True, sharey=True, figsize=(50,20))
            # input images on top row, reconstructions on bottom
            for noisy_imgs, row in zip([images_np, output_np, target_np], axes):
                for img, ax in zip(noisy_imgs, row):
                    ax.imshow(utils.preplot(utils.normalize(np.squeeze(img))))
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
            plt.savefig(RESULTS_TRAIN_PATH + str(epoch) + '.png')
            #plt.show()
            plt.close()
            utils.plot_losses(losses[:], name='epoch_loss', RESULTS_DIR=RESULTS_DIR, method=method)
            utils.plot_losses(iter_losses[:], name='iter_loss', RESULTS_DIR=RESULTS_DIR, method=method)

        # ==============================
        #  SAVE RESULTS & CHECKPOINT 
        # ==============================                
        if epoch % save_freq == 0:
            utils.check_dir([RESULTS_TRAIN_PATH+str(epoch)])
            
            output_out = outputs.detach().cpu().numpy()
            images_out = images.detach().cpu().numpy()
            target_out = target.detach().cpu().numpy()
            
            #print(output_out.shape)
            for i in range (diff_loader.batch_size):
                reconstruction = Image.fromarray((utils.normalize(utils.preplot(np.squeeze(output_out[i,:,:,:]))) * 255).astype('uint8'))
                imgs = Image.fromarray((utils.normalize(utils.preplot(np.squeeze(images_out[i,:,:,:]))) * 255).astype('uint8'))
                gt = Image.fromarray((utils.normalize(utils.preplot(np.squeeze(target_out[i,:,:,:]))) * 255).astype('uint8'))
                
                reconstruction.save(RESULTS_TRAIN_PATH+str(epoch)+'/'+str(i)+'.png')
                imgs.save(RESULTS_TRAIN_PATH+'input/'+str(i)+'.png')
                gt.save(RESULTS_TRAIN_PATH+'gt/'+str(i)+'.png')
                    
            CHECKPOINT_PATH = CHECKPOINT_DIR + str(epoch) + '.pt'
            if ngpu > 1:
                netG_state = model_G.module.state_dict()
                netD_state = model_D.module.state_dict()
            else:
                netG_state = model_G.state_dict()
                netD_state = model_D.state_dict()
            checkpoint_state = {'netG': netG_state,
                                'netD': netD_state,
                                'optimizerG': optimizer_G.state_dict(),
                                'optimizerD': optimizer_D.state_dict(),
                                'schedulerG': scheduler_G.state_dict(),
                                'schedulerD': scheduler_D.state_dict(),
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

    return model_G
    
    
