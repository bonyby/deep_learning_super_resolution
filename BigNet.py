import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch.nn.functional as F
import math
import random
import cv2
import PIL
import time

from torchsummary import summary
from torchvision import datasets
from torchvision import transforms
from torch import nn
from torch import optim
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from PIL import Image

class BigNet(nn.Module):
    def __init__(self):
        super(BigNet, self).__init__()

        scale = 3
        num_residual_blocks = 16
        residual_block_channels = 64

        # First conv layer
        self.conv1 = nn.Conv2d(3, residual_block_channels, (3,3), padding=1, stride=1)

        # Residual blocks
        residual_blocks = [ResidualBlock(residual_block_channels) for _ in range(num_residual_blocks)]
        self.resBlocks = nn.Sequential(*residual_blocks)

        # Post-Residual conv layer
        self.conv2 = nn.Conv2d(residual_block_channels, residual_block_channels, (3,3), padding=1, stride=1)

        # Upsampler
        self.upsampler = Upsampler(scale, residual_block_channels)

    def forward(self, x):
        # First conv + identity for global residual learning
        x = self.conv1(x)
        identity = x
        
        # Residual blocks
        x = self.resBlocks(x)
        
        # Post-Residual conv layer + skip connection
        x = self.conv2(x)
        x = torch.add(x, identity)
        
        # Upsampling
        x = self.upsampler(x)

        return x


   

#Residual Block consists of (Conv, ReLU, Conv), with a skip connection from before the first conv to after the last
#All kernels are (3,3) with a padding of 1
class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()

        structure = []
        structure.append(nn.Conv2d(num_channels, num_channels, (3,3), padding=1, stride=1))
        structure.append(nn.ReLU())
        structure.append(nn.Conv2d(num_channels, num_channels, (3,3), padding=1, stride=1))

        self.body = nn.Sequential(*structure)

    def forward(self, x):
        #Saving the identity for skip-connection
        identity = x
        
        #Feeding through the body
        x = self.body(x)

        #Skip-connection
        x = torch.add(x, identity)
        return x

class Upsampler(nn.Module):
    def __init__(self, scale, num_channels):
        super(Upsampler, self).__init__()
        
        structure = []
        structure.append(nn.Conv2d(num_channels, num_channels * scale**2, (3,3), padding=1, stride=1))
        structure.append(nn.PixelShuffle(scale))
        structure.append(nn.Conv2d(num_channels, 3, (3,3), padding=1, stride=1))
        
        self.body = nn.Sequential(*structure)

    def forward(self, x):
        return self.body(x)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #output features is in_channels + 16
        return torch.cat([x, self.relu(self.conv(x))], dim=1)

class DenseEndLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseEndLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #output features is in_channels + 16
        return self.relu(self.conv(x))



class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        output_features_per_DenseLayer = out_channels
        num_denseLayers = 2
        
        self.block = [nn.Conv2d(in_channels, output_features_per_DenseLayer, kernel_size=3, padding=1)]

        self.block.append(DenseLayer(output_features_per_DenseLayer, output_features_per_DenseLayer))
        self.block.append(DenseLayer(output_features_per_DenseLayer*2, output_features_per_DenseLayer))
        self.block.append(DenseEndLayer(output_features_per_DenseLayer*3, output_features_per_DenseLayer))
        
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)

#All kernels are (3,3) with a padding of 1


class UNetAttentionDeepDense(nn.Module):
    def __init__(self):
        super(UNetAttentionDeepDense, self).__init__()
        residual_block_channels = 64

        self.upsample = nn.Upsample(scale_factor=3, mode='bicubic')
        prepool = []

        prepool.append(nn.Conv2d(3, 64, (3,3), padding=1, stride=1))
        prepool.append(nn.ReLU())
        prepool.append(nn.Conv2d(64, 64, (3,3), padding=1, stride=1)) #skip1 is 32 layers

        self.preConv = nn.Sequential(*prepool)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.resBlock1 = DenseBlock(64, 128) # skip2 is 64 layers
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.resBlock2 = DenseBlock(128,256) #Lowpoint
        
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.resBlock3 = DenseBlock(256, 512)

        self.deconv1 = nn.ConvTranspose2d(512, 256, (2, 2), padding=0, stride=2)
        self.resBlock4 = DenseBlock(512, 256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, (2, 2), padding=0, stride=2)

        self.resBlock5 = DenseBlock(256, 128)

        self.deconv3 = nn.ConvTranspose2d(128, 64, (2, 2), padding=0, stride=2)
        
        postpool = []
        postpool.append(nn.Conv2d(128, 64, (3,3), padding=1, stride=1))
        postpool.append(nn.ReLU())
        postpool.append(nn.Conv2d(64, 32, (3,3), padding=1, stride=1))
        postpool.append(nn.ReLU())

        self.sigmoidconv = nn.Conv2d(32, 3, (3,3), padding=1, stride=1)
        
        self.postConv = nn.Sequential(*postpool)


        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        #Saving the identity for skip-connection
        x = self.upsample(x)
        x = self.preConv(x)
        skip1 = x
        x = self.max_pool(x)
        x = self.resBlock1(x)
        skip2 = x
        x = self.avg_pool1(x)
        x = self.resBlock2(x)
        skip3 = x
        x = self.avg_pool2(x)
        x = self.resBlock3(x)
        #print("x shape:", x.size())
        x = self.deconv1(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.resBlock4(x)
        x = self.deconv2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.resBlock5(x)
        x = self.deconv3(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.postConv(x)
        x = self.sigmoidconv(x)
        x = self.sigmoid(x)
        return x





class UNetAttentionResBlocks(nn.Module):
    def __init__(self):
        super(UNetAttentionResBlocks, self).__init__()

        self.upsample = nn.Upsample(scale_factor=3, mode='bicubic')
        prepool = []

        prepool.append(nn.Conv2d(3, 64, (3,3), padding=1, stride=1))
        prepool.append(nn.ReLU())
        prepool.append(nn.Conv2d(64, 64, (3,3), padding=1, stride=1)) #skip1 is 64 layers

        self.preConv = nn.Sequential(*prepool)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.resAdatper1 = nn.Conv2d(64, 128, (3,3), padding=1, stride=1)
        self.bnorm1 = nn.BatchNorm2d(num_features=128)
        self.reluAdapt1 = nn.ReLU()
        self.resBlock1 = ResidualBlock(128) # skip2 is 64 layers
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.resAdatper2 = nn.Conv2d(128, 256, (3,3), padding=1, stride=1)
        self.bnorm2 = nn.BatchNorm2d(num_features=256)
        self.reluAdapt2 = nn.ReLU()
        self.resBlock2 = ResidualBlock(256)
        
        self.deconv1 = nn.ConvTranspose2d(256, 128, (2, 2), padding=0, stride=2)
        self.skip2catch = nn.Conv2d(256, 128, (3,3), padding=1, stride=1)
        self.bnorm3 = nn.BatchNorm2d(num_features=128)
        self.skip2ReLU = nn.ReLU()
        self.resBlock3 = ResidualBlock(128)

        self.deconv2 = nn.ConvTranspose2d(128, 64, (2, 2), padding=0, stride=2)
        
        self.postpool1 = nn.Conv2d(128, 64, (3,3), padding=1, stride=1)
        self.postpool2 = nn.ReLU()
        self.postpool22 = nn.Conv2d(64, 64, (3,3), padding=1, stride=1)
        self.postpool23 = nn.ReLU()
        self.postpool3 = nn.Conv2d(64, 1, (3,3), padding=1, stride=1)


        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        #Saving the identity for skip-connection
        x = self.upsample(x)
        x = self.preConv(x)
        skip1 = x
        x = self.max_pool(x)
        x = self.resAdatper1(x)
        x = self.bnorm1(x)
        x = self.reluAdapt1(x)
        x = self.resBlock1(x)
        skip2 = x
        x = self.avg_pool1(x)
        x = self.resAdatper2(x)
        x = self.bnorm2(x)
        x = self.reluAdapt2(x)
        x = self.resBlock2(x)
        x = self.deconv1(x)
        #print("x shape:", x.size(), "skip2 shape", skip2.size())
        x = torch.cat([x, skip2], dim=1)
        #print("x shape:", x.size())
        x = self.skip2catch(x)
        x = self.bnorm3(x)
        x = self.skip2ReLU(x)
        x = self.resBlock3(x)
        x = self.deconv2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.postpool1(x)
        x = self.postpool2(x)
        x = self.postpool22(x)
        x = self.postpool23(x)
        x = self.postpool3(x)
        x = self.sigmoid(-x)
        return x






class BigAttentionNet(nn.Module):
    def __init__(self):
        super(BigAttentionNet, self).__init__()
        self.bignet = BigNet()
        self.attention = UNetAttentionResBlocks()

        self.upsampler = nn.Upsample(scale_factor = 3, mode='bicubic')

    
    def forward(self, x):
        #print("this is x:", x.size())
        scaledX = self.upsampler(x)
        #print("this is x", x.size())
        
        bignetOutput = self.bignet(x)
        attentionOutput = self.attention(x)
        # print("bignetoutput shape:", bignetOutput.size())
        # print("attentionOutput", attentionOutput.size())
        # print("scaled x shape:", scaledX.size())
        # print("x shape:", x.size())

        combination = torch.mul(bignetOutput, attentionOutput)
        result = torch.add(combination, scaledX)
        return result



















# class UNetAttention(nn.Module):
#     def __init__(self):
#         super(UNetAttention, self).__init__()
#         residual_block_channels = 64

#         self.upsample = nn.Upsample(scale_factor=3, mode='bicubic')
#         prepool = []

#         prepool.append(nn.Conv2d(3, 32, (3,3), padding=1, stride=1))
#         prepool.append(nn.ReLU())
#         prepool.append(nn.Conv2d(32, 32, (3,3), padding=1, stride=1)) #skip1 is 32 layers

#         self.preConv = nn.Sequential(*prepool)
#         self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.resBlock1 = DenseBlock(32) # skip2 is 64 layers
#         self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

#         self.resBlock2 = DenseBlock(64) #Lowpoint
        
#         self.deconv1 = nn.ConvTranspose2d(96, 96, (2, 2), padding=0, stride=2)
#         self.resBlock3 = DenseBlock(160)

#         self.deconv2 = nn.ConvTranspose2d(192, 192, (2, 2), padding=0, stride=2)
        
#         self.postpool1 = nn.Conv2d(224, 32, (3,3), padding=1, stride=1)
#         self.postpool2 = nn.ReLU()
#         self.postpool3 = nn.Conv2d(32, 1, (3,3), padding=1, stride=1)


#         self.sigmoid = nn.Sigmoid()


#     def forward(self, x):
#         #Saving the identity for skip-connection
#         x = self.upsample(x)
#         x = self.preConv(x)
#         skip1 = x
#         x = self.max_pool(x)
#         x = self.resBlock1(x)
#         skip2 = x
#         x = self.avg_pool1(x)
#         x = self.resBlock2(x)
#         x = self.deconv1(x)
#         #print("x shape:", x.size(), "skip2 shape", skip2.size())
#         x = torch.cat([x, skip2], dim=1)
#         #print("x shape:", x.size())
#         x = self.resBlock3(x)
#         x = self.deconv2(x)
#         x = torch.cat([x, skip1], dim=1)
#         x = self.postpool1(x)
#         x = self.postpool2(x)
#         x = self.postpool3(x)
#         x = self.sigmoid(x)
#         return x

