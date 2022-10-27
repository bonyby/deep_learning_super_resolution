import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch.nn.functional as F

from torchsummary import summary
from torchvision import datasets
from torchvision import transforms
from torch import nn
from torch import optim
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from PIL import Image


def get_data():
    print("wallah")
    # Get data here


#
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()  # Sets the mode of the model to training
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()  # Sets the mode of the model to evaluation
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print("Epoch: " + str(epoch) + " Val loss: " + str(val_loss))

def MSE(input, target):
    # Crop the target to the input size - the input loses some of it's size as the model doesn't use any padding
    crop_transform = transforms.CenterCrop(input.size(dim=2))
    tgt = crop_transform(target)
    # print("Size of tgt: " + str(tgt.size()))
    # print("Size of input: " + str(input.size()))
    
    # loss = 0
    # bs = input.size(dim = 0)
    # for i in range(bs):
    #     loss += MSE_Single(input[i], tgt[i])

    # return loss / bs

    mse = nn.MSELoss()
    return mse(input, tgt)

def MSE_Single(input, target):
    # Calculate MSE
    inputNp = input.clone()
    inputNp = inputNp.detach().numpy()
    targetNp = target.clone()
    targetNp = targetNp.detach().numpy()

    loss = np.sum((inputNp - targetNp) ** 2)
    loss /= (input.size(dim=0) * input.size(dim=1) * input.size(dim=2))
    return loss

def SRCNN(scale):
    # This model has 8129 parameters when channels = 1 (20099 when channels = 3) including bias, 8032 without - which is as in the article.
    channels = 3
    n1 = 64
    n2 = 32
    f1 = 9
    f2 = 1
    f3 = 5

    model = nn.Sequential(
        # Preprocessing with bicubic interpolation
        nn.Upsample(scale_factor=scale, mode='bicubic', align_corners= False),

        # Model
        nn.Conv2d(channels, n1, (f1, f1)),
        nn.ReLU(),
        nn.Conv2d(n1, n2, (f2, f2)),
        nn.ReLU(),
        nn.Conv2d(n2, channels, (f3, f3)),
        nn.ReLU()
    )

    return model


def reset_parameters(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1) # Why 1?
            torch.nn.init.constant_(m.bias, 0) # Why 0?
        elif isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)



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
        img = pil_tensor_converter(Image.open(f))
        images.append(img)

        counter -= 1

    # Blur the input image (gaussian) and downsample
    downsampledStack = torch.stack(images)
    blurredImages = gaussian_transform(downsampledStack)
    downsampledImages = F.interpolate(blurredImages, size=(blurredImages[0].size(dim=1) // downscale_factor, blurredImages[0].size(dim=2) // downscale_factor), mode="nearest")

    imgStack = torch.stack(images)
    return TensorDataset(downsampledImages, imgStack)

def get_image_dataloaders(path, n=-1, bs=8, downscale_factor = 3):
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
    train_dl = DataLoader(train_dataset, batch_size=bs)
    test_dl = DataLoader(test_dataset, batch_size=bs)
    val_dl = DataLoader(val_dataset, batch_size=bs)

    return train_dl, test_dl, val_dl

def show_tensor_as_image(tensor):
    if tensor.ndimension() == 4:
        tensor = tensor[0]

    tensor_pil_converter = transforms.ToPILImage()
    tensor_pil_converter(tensor).show()

# ------------------- CODE -------------------

bs = 64
train_dl, test_dl, val_dl = get_image_dataloaders("./Datasets/T91/T91_HR_Patches", 5000, bs = bs, downscale_factor = 3)

lr = 0.0001
model = SRCNN(scale=3)
reset_parameters(model)
opt = optim.SGD(model.parameters(), lr = lr, momentum=0.9)

epochs = 15
loss_func = MSE
fit(epochs, model, loss_func, opt, train_dl, val_dl)

for xb, yb in train_dl:
    show_tensor_as_image(model(xb[None, 0]))
    show_tensor_as_image(yb[0])
    break

