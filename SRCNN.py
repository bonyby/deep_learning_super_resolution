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


#https://github.com/yjn870/SRCNN-pytorch/blob/master/train.py
#https://medium.com/coinmonks/review-srcnn-super-resolution-3cb3a4f67a7c


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


# Function handle that returns an optimizer
def basic_SGD(model,lr=0.0001, momentum=0.9):
    return optim.SGD(model.parameters(), lr=lr,momentum=momentum)

# Function handle that updates the learning rate
# (note this is a dummy implementation that does nothing)
def base_lr_scheduler(t,T,lr):
  return lr





def fit2(model,
        trainset, 
        testset,
        loss_func,
        accuracy,
        opt_func,
        batch_prints=False,
        lr_scheduler=base_lr_scheduler,
        bs=256,
        epochs=1, 
        batches_per_epoch=None, # Default: Use entire training set
        show_summary=True):

  # Set up data loaders
  if batches_per_epoch == None:
    # Use all images
    train_dl = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                          shuffle=True, num_workers=2)
    valid_dl = torch.utils.data.DataLoader(testset, batch_size=bs,
                                         shuffle=False, num_workers=2)
    batches_per_epoch = len(train_dl)
  else:
    print("Yo im here")
    # Only use a subset of the data
    subset_indices = list(range(batches_per_epoch*bs))
    train_dl = torch.utils.data.DataLoader(trainset, batch_size=bs, sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices), num_workers=2)
    print("HERRREREREROWAFA")
    # Use one fourth for validation
    subset_indices = list(range(int(np.ceil(batches_per_epoch/4))*bs))
    valid_dl = torch.utils.data.DataLoader(testset, batch_size=bs, sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices), num_workers=2)

  # Initialize optimizer
  opt = opt_func(model)

  # For book keeping
  train_loss_history = []
  valid_loss_history = []
  plot_time_train = []
  plot_time_valid = []

  # Index of current batch
  t = 1

  # Total number of batches
  T = batches_per_epoch * epochs
  
  print('Epochs:',epochs,'Batches per epoch:',batches_per_epoch,'Total number of batches',T)
  
  # Get initial validation loss and accuracy
  model.eval()
  with torch.no_grad():
    valid_acc = sum(accuracy(model(xb.cuda()), yb.cuda()) for xb, yb in valid_dl) / len(valid_dl)
    valid_loss = sum(loss_func(model(xb.cuda()), yb.cuda()) for xb, yb in valid_dl) / len(valid_dl)
    valid_loss_history.append(valid_loss.detach().cpu().numpy())
    plot_time_valid.append(t)

  # Train
  for epoch in range(epochs):
    model.train() # Train mode
    for xb, yb in train_dl:

      # Update learning rate
      opt.param_groups[0]['lr'] = lr_scheduler(t,T,lr=opt.param_groups[0]['lr'])

      # Forward prop
      pred = model(xb.cuda())
      loss = loss_func(pred, yb.cuda())

      # Book keeping
      train_loss_history.append(loss.detach().cpu().numpy())
      plot_time_train.append(t)
      t += 1

      # Backward prop (calculate gradient)
      loss.backward()

      # Update model parameters
      opt.step()
      opt.zero_grad()

      # Validation loss and accuracy
      if t % 10 == 0 and batch_prints:    # print every 10 mini-batches
        model.eval() # Test mode
        with torch.no_grad():
            valid_acc = sum(accuracy(model(xb.cuda()), yb.cuda()) for xb, yb in valid_dl) / len(valid_dl)
            valid_loss = sum(loss_func(model(xb.cuda()), yb.cuda()) for xb, yb in valid_dl) / len(valid_dl)
            valid_loss_history.append(valid_loss.detach().cpu().numpy())
            plot_time_valid.append(t-1)
            print('t',t,', lr',opt.param_groups[0]['lr'],'train loss',loss.detach().cpu().numpy(), 'val loss',valid_loss.detach().cpu().numpy(),'val accuracy (PSNR)', valid_acc.detach().cpu().numpy())
        model.train() # Back to train mode

        #At the end of an epoch:
      if t % bs-1 == 0: #-1 or not?
        model.eval() # Test mode
        with torch.no_grad():
            valid_acc = sum(accuracy(model(xb.cuda()), yb.cuda()) for xb, yb in valid_dl) / len(valid_dl)
            valid_loss = sum(loss_func(model(xb.cuda()), yb.cuda()) for xb, yb in valid_dl) / len(valid_dl)
            valid_loss_history.append(valid_loss.detach().cpu().numpy())
            plot_time_valid.append(t-1)
            print('Epoch: ',epoch,', lr',opt.param_groups[0]['lr'],'train loss',loss.detach().cpu().numpy(), 'val loss',valid_loss.detach().cpu().numpy(),'val accuracy', valid_acc.detach().cpu().numpy())
        model.train() # Back to train mode
    


  # Summary
  if show_summary:
    plt.figure()
    lines = []
    labels = []
    l, = plt.plot(plot_time_train,train_loss_history)
    lines.append(l)
    labels.append('Training')
    l, = plt.plot(plot_time_valid,valid_loss_history)
    lines.append(l)
    labels.append('Validation')  
    plt.title('Loss')
    plt.legend(lines, labels, loc=(1, 0), prop=dict(size=14))
    plt.show()

  return train_loss_history








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
    return mse(input, target)

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
        #nn.ReLU() #this should not be here - it is the last layer!
    )

    return model


class ModularSRCNN(nn.Module):
   
    def __init__(self):
         # This model has 8129 parameters when channels = 1 (20099 when channels = 3) including bias, 8032 without - which is as in the article.
        channels = 3
        n1 = 64
        n2 = 32
        f1 = 9
        f2 = 1
        f3 = 5

        super(ModularSRCNN, self).__init__()

        self.conv1 = nn.Conv2d(channels, n1, (f1, f1))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(n1, n2, (f2, f2))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(n2, channels, (f3, f3))
        #self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        #x = self.relu3(x)
        return x

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

def get_image_dataloaders(path, n=-1, bs=8, downscale_factor = 3):
    train_dataset, test_dataset, val_dataset = get_image_sets(path, n, downscale_factor)
    train_dl = DataLoader(train_dataset, batch_size=bs)
    test_dl = DataLoader(test_dataset, batch_size=bs)
    val_dl = DataLoader(val_dataset, batch_size=bs)

    return train_dl, test_dl, val_dl

def show_tensor_as_image(tensor):
    if tensor.ndimension() == 4:
        tensor = tensor[0]

    tensor_pil_converter = transforms.ToPILImage()
    tensor_pil_converter(tensor).show()

def show_cuda_tensor_as_image(tensor):
    if tensor.ndimension() == 4:
        tensor = tensor[0]

    tensor_pil_converter = transforms.ToPILImage()
    tensor_pil_converter(tensor).show()

def PSNRaccuracy(scores, yb, max_val=1):
    crop_transform = transforms.CenterCrop(scores.size(dim=2))
    yb_cropped = crop_transform(yb)

    diff = scores - yb_cropped
    diff_flat = torch.flatten(diff, start_dim = 1)
    
    rmse = torch.sqrt(torch.mean(torch.square(diff_flat)))
    # 10 * log_10(MAX_I^2/MSE)
    # divres = 1/mse
    # logres = torch.log10(divres)
    # mulres = 10 * logres
    # return mulres

    return Tensor([20 * math.log(max_val/rmse, 10)])

    # return Tensor([0.5])
    #score2prob = nn.Softmax(dim=1)
    #preds = torch.argmax(score2prob(scores), dim=1)
    #return (preds == yb).float().mean()

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


def get_preprocessed_image_sets(path_low, path_high, n=-1): #This function works on preprocessed data
    dataset = get_tensor_images_high_low(path_low, path_high, n)
    n = len(dataset)
    print("I'm here now 5")
    # Calculate the sizes of the data subsets
    # We need to check the size difference to correct for rounding errors
    trainSize = n*8//10
    testSize = n//10
    valSize = n//10
    sizeDiff = trainSize + testSize + valSize - n
    trainSize -= sizeDiff # correct for rounding errors
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [trainSize, testSize, valSize])
    return train_dataset, test_dataset, val_dataset



def get_tensor_images_high_low(path_low, path_high, n=-1): #This function works on preprocessed data
    images_low = []
    images_high = []
    pil_tensor_converter = transforms.ToTensor()
    print("I'm here now 1")
    # Convert all images on the given path to tensors
    counter_low = n
    for f in sorted(glob.iglob(path_low + "/*")):
        # Break if n images appended (if n = -1 - aka get all images - this never occurs)
        if counter_low == 0:
            break
        if counter_low == -1000:
            name = (f.split("\\")[1]).split(".")[0]
            print("The name: " + name)
        im = Image.open(f)
        #print("Image mode: " + str(im.mode)) - The images are RGB
        img = pil_tensor_converter(im)
        images_low.append(img)
        counter_low -= 1


    print("I'm here now 2")
    counter_high= n
    for f in sorted(glob.iglob(path_high + "/*")):
        # Break if n images appended (if n = -1 - aka get all images - this never occurs)
        if counter_high == 0:
            break
        if counter_high == -1000:
            name = (f.split("\\")[1]).split(".")[0]
            print("The name: " + name)
        im = Image.open(f)
        #print("Image mode: " + str(im.mode)) - The images are RGB
        img = pil_tensor_converter(im)
        images_high.append(img)

        counter_high -= 1
    print("I'm here now 3")
    low_res_stack = torch.stack(images_low)
    high_res_stack = torch.stack(images_high)
    
    print("I'm here now 4")
    return TensorDataset(low_res_stack, high_res_stack)

# ------------------- CODE -------------------
def main():
    bs = 128
    train_dl, test_dl, val_dl = get_image_dataloaders("./Datasets/T91/T91_HR_Patches", 1000, bs = bs, downscale_factor = 3)

    
    training, validation, testing  = get_preprocessed_image_sets("./Datasets/T91/T91_Upscaled_Patches", "./Datasets/T91/T91_HR_Patches")


    #training, validation, testing = get_image_sets("./Datasets/T91/T91_HR_Patches", downscale_factor = 3)
    lr = 0.0001

    model = ModularSRCNN().cuda()

    reset_parameters(model)


    epochs = 50
    loss_func = MSE



    fit2(model, training, validation, loss_func, PSNRaccuracy, basic_SGD, bs=bs, epochs=epochs)
    counter = 4
    for xb, yb in train_dl:
        if(counter == 0):
            break
        #print(xb.cuda()[None,0].max())
        model.eval()
        result = model(xb.cuda()[None, 0])
        show_tensor_as_image(result)
        print("result size: " + str(result.size()))
        crop_transform = transforms.CenterCrop(result.size(dim=2))
        tgt = crop_transform(yb.cuda()[0])
        print("tgt size: " + str(tgt.size()))

        show_tensor_as_image(tgt)
        counter -= 1


if __name__ == "__main__":
    main()
    
    
    
    '''
    def fit(model,
            trainset, 
            testset,
            loss_func,
            accuracy,
            opt_func,
            batch_prints=False,
            lr_scheduler=base_lr_scheduler,
            bs=256,
            epochs=1, 
            batches_per_epoch=None, # Default: Use entire training set
            show_summary=True):
            '''