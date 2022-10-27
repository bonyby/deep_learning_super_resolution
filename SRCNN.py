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

        print(epoch, val_loss)

# Currently between two images


def MSE(input, target):
    loss = np.sum((input - target) ** 2)
    loss /= (input.shape[0] * input.shape[1])
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
        nn.Upsample(scale_factor=scale, mode='bicubic'),

        # Model
        nn.Conv2d(channels, n1, (f1, f1)),
        nn.ReLU(),
        nn.Conv2d(n1, n2, (f2, f2)),
        nn.ReLU(),
        nn.Conv2d(n2, channels, (f3, f3)),
        nn.ReLU()
    )

    return model


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
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(hr_patches, [n*8//10, n//10, n//10])
    train_dl = DataLoader(train_dataset, batch_size=bs)
    test_dl = DataLoader(test_dataset, batch_size=bs)
    val_dl = DataLoader(val_dataset, batch_size=bs)

    return train_dl, test_dl, val_dl

def show_tensor_as_image(tensor):
    tensor_pil_converter = transforms.ToPILImage()
    tensor_pil_converter(tensor).show()


# ------------------- CODE -------------------
# dataset = datasets.StanfordCars(root="/.", download=True)
# dataset = datasets.Places365(root="/.", download=True)
bs = 8
train_dl, test_dl, val_dl = get_image_dataloaders("./Datasets/T91/T91_HR_Patches", 160, bs, 3)

# for xb, yb in train_dl:
#     print("xb[0] size: " + str(xb[0].size()))
#     print("yb[0] size: " + str(yb[0].size()))

#     show_tensor_as_image(xb[0])
#     show_tensor_as_image(yb[0])

#     break

# img, label = dataset[0]
# print(img.size)

# train = "" # not defined yet
# val = "" # not defined yet

# lr = 0.01
# model = SRCNN(scale=3)
# opt = optim.SGD(model.parameters(), lr = lr, momentum=0.9)

# epochs = 10
# loss_func = MSE
# fit(epochs, model, loss_func, opt, train, val)
