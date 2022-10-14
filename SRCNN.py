import torch
import numpy as np
import matplotlib as plt

from torchsummary import summary
from torchvision import datasets
from torch import nn
from torch import optim
from torch.utils.data import DataLoader



def get_data():
    print("wallah")
    #Get data here


    


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
        model.train() #Sets the mode of the model to training
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval() #Sets the mode of the model to evaluation
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
    #This model has 8129 parameters when channels = 1 (20099 when channels = 3) including bias, 8032 without - which is as in the article.
    channels = 3
    n1 = 64
    n2 = 32
    f1 = 9
    f2 = 1
    f3 = 5

    model = nn.Sequential(
        #Preprocessing with bicubic interpolation
        nn  .Upsample(scale_factor=scale, mode='bicubic'),

        #Model
        nn.Conv2d(channels, n1, (f1, f1)), 
        nn.ReLU(),
        nn.Conv2d(n1, n2, (f2, f2)),
        nn.ReLU(),
        nn.Conv2d(n2, channels, (f3, f3)),
        nn.ReLU()
    )

    return model


#------------------- CODE -------------------
# dataset = datasets.StanfordCars(root="/.", download=True)
dataset = datasets.Places365(root="/.", download=True)
img, label = dataset[0]
print(img.size)

# train = "" # not defined yet
# val = "" # not defined yet

# lr = 0.01
# model = SRCNN(scale=3)
# opt = optim.SGD(model.parameters(), lr = lr, momentum=0.9)

# epochs = 10
# loss_func = MSE
# fit(epochs, model, loss_func, opt, train, val)