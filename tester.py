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

from scripts import utils, models
import trainer

# ############################################
#     TEST MODEL
# ############################################
def test(method, model_G, model_D, diff_loader, gt_loader, last_epoch, RESULTS_DIR, device, mode='test'):
    """ 
    Test model - Parameters:
        -model_G:       Model to test.
        -diff_loader:   Test set input images.
        -gt_loader:     Test set ground truth images.
        -mode:          Whether this is a test or validation set. Options: ["test", "val"]
        -last_epoch:    Current training epoch of model.
        -RESULTS_DIR:   Path to results directory.
        -device:        Designates which GPU should be used to load test images.
    """
    with torch.no_grad():
        # device = torch.device("cuda:0")
        model_G = model_G.to(device); model_D.to(device)
        model_G.eval(); model_D.eval()

        # obtain one batch of test images
        dataiter = iter(diff_loader)
        images_test = dataiter.next()[0]
        dataiter = iter(gt_loader)
        target_test = dataiter.next()[0]
      
        # get sample outputs
        output_test = model_G(images_test.to(device))
        # calculate the loss
        G_mse_loss = 1 * trainer.loss_MSE(output_test, target_test.to(device))
        #G_mse_loss.backward(retain_graph=True)
        # the "target" is still the original, not-noisy images
        pred_fake = model_D(output_test)
        if method == 'WGAN':
            G_adv_loss = 0.01 * pred_fake.mean()
            #G_adv_loss.backward(mone)
        else:
            G_adv_loss = 0.01 * trainer.loss_disc_gt(pred_fake)
        loss_G = G_mse_loss + G_adv_loss
    
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
                
        RESULTS_TEST_PATH = RESULTS_DIR + mode + '/'
        utils.check_dir([RESULTS_TEST_PATH, RESULTS_TEST_PATH+str(last_epoch), RESULTS_TEST_PATH+'input/', RESULTS_TEST_PATH+'gt/'])

        for i in range (diff_loader.batch_size):
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
    return loss_G.item()