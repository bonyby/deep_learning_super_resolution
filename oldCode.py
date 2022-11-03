import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch.nn.functional as F
import math

from torchsummary import summary
from torchvision import datasets
from torchvision import transforms
from torch import nn
from torch import optim
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from PIL import Image


def get_image_dataloaders(path, n=-1, bs=8, downscale_factor = 3):
    train_dataset, test_dataset, val_dataset = get_image_sets(path, n, downscale_factor)
    train_dl = DataLoader(train_dataset, batch_size=bs)
    test_dl = DataLoader(test_dataset, batch_size=bs)
    val_dl = DataLoader(val_dataset, batch_size=bs)

    return train_dl, test_dl, val_dl




def get_tensor_images(path, n=-1, downscale_factor = 3):
    images = []
    downsampledImages = []
    pil_tensor_converter = transforms.ToTensor()
    gaussian_transform = transforms.GaussianBlur((3,3))

    # Convert all images on the given path to tensors
    counter = n
    for f in glob.iglob(path + "/*"):
        # Break if n images appended (if n = -1 - aka get all images - this never occurs)
        if counter == 0:
            break
        im = Image.open(f)
        #print("Image mode: " + str(im.mode)) - The images are RGB
        img = pil_tensor_converter(im)
        images.append(img)

        counter -= 1

    # Blur the input image (gaussian) and downsample
    downsampledStack = torch.stack(images)
    blurredImages = gaussian_transform(downsampledStack)
    downsampledImages = F.interpolate(blurredImages, size=(blurredImages[0].size(dim=1) // downscale_factor, blurredImages[0].size(dim=2) // downscale_factor), mode="nearest")

    imgStack = torch.stack(images)
    return TensorDataset(downsampledImages, imgStack)


def get_image_sets(path, n=-1, downscale_factor = 3):
    hr_patches = get_tensor_images(path, n, downscale_factor)
    n = len(hr_patches)
    
    # Calculate the sizes of the data subsets
    # We need to check the size difference to correct for rounding errors
    trainSize = n*8//10
    testSize = n//10
    valSize = n//10
    sizeDiff = trainSize + testSize + valSize - n
    trainSize -= sizeDiff # correct for rounding errors
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(hr_patches, [trainSize, testSize, valSize])
    return train_dataset, test_dataset, val_dataset
