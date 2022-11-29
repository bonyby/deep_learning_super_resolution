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



class ResidualDenseBlock(nn.Module):
    #TODO: Implement
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