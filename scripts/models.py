import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import utils as tvutils
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
#import cv2
import os
import glob

unet = 1

def define_D(netD='basic'):
    if netD == 'basic':
        discriminator = Discriminator()
    elif netD == 'cycle':
        discriminator = Discriminator_Cycle()
    elif netD == 'wgan':
        discriminator = Discriminator_WGAN()
    return discriminator

def define_G(netG='unet'):
    if netG == 'unet':
        generator = UNet((3, 256, 256))
    return generator


        
class LSID(nn.Module):
    def __init__(self, inchannel=1, block_size=2):
        super(LSID, self).__init__()
        self.block_size = block_size

        self.conv1_1 = nn.Conv2d(inchannel, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        #self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        #self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)

        #self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False)
        #self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        #self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        out_channel = 3 * self.block_size * self.block_size
        self.conv10 = nn.Conv2d(32, out_channel, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = x
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        conv2 = x
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        conv3 = x
        x = self.maxpool(x)

        x = self.conv4_1(x)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.lrelu(x)
        conv4 = x
        x = self.maxpool(x)

        #x = self.conv5_1(x)
        #x = self.lrelu(x)
        #x = self.conv5_2(x)
        #x = self.lrelu(x)

        #x = self.up6(x)
        #x = torch.cat((x[:, :, :conv4.size(2), :conv4.size(3)], conv4), 1)
        #x = self.conv6_1(x)
        #x = self.lrelu(x)
        #x = self.conv6_2(x)
        #x = self.lrelu(x)

        x = self.up7(x)
        #x = torch.cat((x[:, :, :conv3.size(2), :conv3.size(3)], conv3), 1)

        x = self.conv7_1(x)
        x = self.lrelu(x)
        x = self.conv7_2(x)
        x = self.lrelu(x)

        x = self.up8(x)
        #x = torch.cat((x[:, :, :conv2.size(2), :conv2.size(3)], conv2), 1)

        x = self.conv8_1(x)
        x = self.lrelu(x)
        x = self.conv8_2(x)
        x = self.lrelu(x)

        x = self.up9(x)
        #x = torch.cat((x[:, :, :conv1.size(2), :conv1.size(3)], conv1), 1)

        x = self.conv9_1(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.lrelu(x)

        x = self.conv10(x)

        depth_to_space_conv = pixel_shuffle(x, upscale_factor=self.block_size, depth_first=True)

        return depth_to_space_conv
BN_EPS = 1e-4
# define the NN architecture
class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  
        # conv layer (depth from 32 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        # transpose layer, a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 3, stride=2)  # kernel_size=3 to get to a 7x7 image output
        # two more transpose layers with a kernel of 2
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # transpose again, output should have a sigmoid appli    
        x = F.sigmoid(self.conv_out(x))
            
                
        return x

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=(3, 3)):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )
    
    def forward(self, x):
        x = self.encode(x)
        x_small = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, x_small
        
class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2
    
        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )
    
    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()
        x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True) #Updated from F.upsample()
        x = torch.cat([x, down_tensor], 1)
        x = self.decode(x)
        return x
        
class UNet(nn.Module):
    def __init__(self, in_shape):
        super(UNet, self).__init__()
        channels, height, width = in_shape
    
        self.down1 = StackEncoder(channels, 24, kernel_size=3) ;# 256
        self.down2 = StackEncoder(24, 64, kernel_size=3)  # 128
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # 64
        self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
        self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16
            
    
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)  # 256
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 512
        self.classify = nn.Conv2d(24, channels, kernel_size=1, bias=True)
    
    
        self.center = nn.Sequential(ConvBnRelu2d(512, 512, kernel_size=3, padding=1))
        #self.center = nn.Sequential(ConvBnRelu2d(256, 256, kernel_size=3, padding=1))
    
    def forward(self, x):
        out = x; 
        down1, out = self.down1(out); 
        down2, out = self.down2(out); 
        down3, out = self.down3(out); 
        down4, out = self.down4(out); 
        down5, out = self.down5(out); 
    
        out = self.center(out)
        out = self.up5(out, down5); 
        out = self.up4(out, down4); 
        out = self.up3(out, down3); 
        out = self.up2(out, down2); 
        out = self.up1(out, down1); 
    
        out = self.classify(out); 
        out = torch.squeeze(out, dim=1); 
        return out
    
class UNet_small(nn.Module):
    def __init__(self, in_shape):
        super(UNet_small, self).__init__()
        channels, height, width = in_shape
    
        self.down1 = StackEncoder(3, 24, kernel_size=3)  # 512
    
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 512
        self.classify = nn.Conv2d(24, 3, kernel_size=1, bias=True)
    
        self.center = nn.Sequential(
            ConvBnRelu2d(24, 24, kernel_size=3, padding=1),
        )
    
    
    def forward(self, x):
        out = x
        down1, out = self.down1(out)
        out = self.center(out)
        out = self.up1(out, down1)
        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out

class UNet_doubleDec(nn.Module):
    def __init__(self, in_shape):
        super(UNet_doubleDec, self).__init__()
        channels, height, width = in_shape
    
        self.down1 = StackEncoder(channels, 24, kernel_size=3) ;# 256
        self.down2 = StackEncoder(24, 64, kernel_size=3)  # 128
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # 64
        self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
        self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16
            
    
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)  # 256
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 512
        self.classify = nn.Conv2d(24, channels, kernel_size=1, bias=True)
    
    
        self.center = nn.Sequential(ConvBnRelu2d(512, 512, kernel_size=3, padding=1))
        #self.center = nn.Sequential(ConvBnRelu2d(256, 256, kernel_size=3, padding=1))
    
    def forward(self, x):
        out = x; 
        down1, out = self.down1(out); 
        down2, out = self.down2(out); 
        down3, out = self.down3(out); 
        down4, out = self.down4(out); 
        down5, out = self.down5(out); 
    
        out = self.center(out)
        outA = self.up5(out, down5);        outB = self.up5(out, down5)
        outA = self.up4(outA, down4);       outB = self.up4(outB, down4)
        outA = self.up3(outA, down3);       outB = self.up4(outB, down3)
        outA = self.up2(outA, down2);       outB = self.up4(outB, down2)
        outA = self.up1(outA, down1);       outB = self.up4(outB, down1)
    
        outA = self.classify(outA);         outB = self.classify(outB)
        outA = torch.squeeze(outA, dim=1);  outB = torch.squeeze(outB, dim=1)
        return outA, outB
    
    
# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)



class Discriminator_Cycle(nn.Module):
    
    def __init__(self, conv_dim=64):
        super(Discriminator_Cycle, self).__init__()

        # Define all convolutional layers
        # Should accept an RGB image as input and output a single value

        # Convolutional layers, increasing in depth
        # first layer has *no* batchnorm
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False) # x, y = 64, depth 64
        self.conv2 = conv(conv_dim, conv_dim*2, 4) # (32, 32, 128)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4) # (16, 16, 256)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4) # (8, 8, 512)
        
        # Classification layer
        self.conv5 = conv(conv_dim*8, 1, 4, stride=1, batch_norm=False)

    def forward(self, x):
        # relu applied to all conv layers but last
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        # last, classification layer
        out = self.conv5(out)
        return out
    
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        #self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64 [3x256x256]
            nn.Conv2d(in_channels=nc, out_channels=ndf,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32 [256x128x128]
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16 [512x64x64]
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8 [1024x32x32]
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4 
            nn.Conv2d(in_channels=ndf * 8, out_channels=1, 
                      kernel_size=4, stride=1, padding=0, bias=False),
#             # input is (nc * 8) x 64 x 64 [2048x16x16]
#             nn.Conv2d(in_channels=ndf * 8, out_channels=ndf * 16,
#                       kernel_size=4, stride=2, padding=1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # input is (nc * 16) x 64 x 64 [4096x8x8]
#             nn.Conv2d(in_channels=ndf * 16, out_channels=ndf * 32,
#                       kernel_size=4, stride=2, padding=1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*32) x 4 x 4 [8192x4x4]
#             nn.Conv2d(in_channels=ndf * 32, out_channels=1, 
#                       kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator_WGAN(nn.Module):
    def __init__(self, DIM=64):
        self.DIM = DIM
        super(Discriminator_WGAN, self).__init__()
        self.main = nn.Sequential(
            # input = 3, output = 64
            nn.Conv2d(in_channels=3, out_channels=DIM,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            # input = 64, output = 128
            nn.Conv2d(in_channels=DIM, out_channels=2 * DIM,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            # input = 128, output = 256
            nn.Conv2d(in_channels=2 * DIM, out_channels=4 * DIM,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self.linear = nn.Linear(4 * 4 * 4 * DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.DIM)
        output = self.linear(output)
        return output
"""
class Discriminator_WGAN(nn.Module):
	def __init__(self, l=0.2):
		super(Discriminator_WGAN, self).__init__()
		self.net = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1),
			nn.LeakyReLU(l),

			nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(l),

			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.LeakyReLU(l),

			nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(l),

			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.LeakyReLU(l),

			nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(l),

			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.LeakyReLU(l),

			nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(l),

			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(512, 1024, kernel_size=1),
			nn.LeakyReLU(l),
			nn.Conv2d(1024, 1, kernel_size=1)
		)

	def forward(self, x): 
		#print ('D input size :' +  str(x.size()))
		y = self.net(x)
		#print ('D output size :' +  str(y.size()))
		return y.view(y.size()[0])
  """