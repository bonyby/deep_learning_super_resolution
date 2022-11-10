import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch.nn.functional as F
import math
import random
import cv2

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


# https://github.com/yjn870/SRCNN-pytorch/blob/master/train.py
# https://medium.com/coinmonks/review-srcnn-super-resolution-3cb3a4f67a7c

#

class ImageDataset(Dataset):
    def __init__(self, imagesList_lr, imagesList_hr):
        self.imagesList_lr = imagesList_lr
        self.imagesList_hr = imagesList_hr

    def __len__(self):
        return len(self.imagesList_lr)

    def __getitem__(self, index):
        return (
            self.imagesList_lr[index],
            self.imagesList_hr[index]
        )


def loss_batch(model, loss_func, xb, yb, opt=None):
    predictions_unclamped = model(xb.cuda())
    loss = loss_func(predictions_unclamped, yb.cuda())
    psnr = PSNRaccuracy(predictions_unclamped.clamp(0, 1), yb.cuda())

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb), psnr


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    T = len(train_dl)
    # Total number of batches
    I = len(train_dl.dataset)

    print('Epochs:', epochs, 'Total number of batches',
          T, 'Total number of elements', I)

    psnr_history = []
    t_counter = 0
    t = []

    for epoch in range(epochs):
        model.train()  # Sets the mode of the model to training
        # for xb, yb in train_dl:
        #    loss_batch(model, loss_func, xb, yb, opt)
        loss_losses, loss_nums, loss_psnrs = zip(
            *[loss_batch(model, loss_func, xb, yb, opt) for xb, yb in train_dl]
        )

        train_loss = np.sum(np.multiply(
            loss_losses, loss_nums)) / np.sum(loss_nums)
        # average PSNR per batch
        train_psnr = np.sum(loss_psnrs) / len(loss_psnrs)

        model.eval()  # Sets the mode of the model to evaluation
        with torch.no_grad():
            losses, nums, val_psnrs = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        val_psnr = np.sum(val_psnrs) / len(val_psnrs)  # average PSNR per batch

        psnr_history.append(val_psnr)
        t_counter += 1
        t.append(t_counter)

        # print("Epoch: " + str(epoch) + " Val loss: " + str(val_loss) + " Val PSNR: " + str(val_psnr))
        # print("Epoch: " + str(epoch) + " Train loss: " + str(train_loss) + " Val loss: " + str(val_loss) + " Val PSNR: " + str(val_psnr))
        print("Epoch: " + str(epoch) + " Train loss: " + str(train_loss) + " Train PSNR: " +
              str(train_psnr) + " Val loss: " + str(val_loss) + " Val PSNR: " + str(val_psnr))

        if epoch % 100 == 0 and epoch != 0:
            # Save the model
            print("Saving model weights...")
            torch.save(model.state_dict(), "./Models/SRCNN.pth")

    plt.figure()
    lines = []
    labels = []
    # l, = plt.plot(plot_time_train,train_loss_history)
    # lines.append(l)
    # labels.append('Training')
    # l, = plt.plot(plot_time_valid,valid_loss_history)
    # lines.append(l)
    # labels.append('Validation')
    l, = plt.plot(t, psnr_history)
    lines.append(l)
    labels.append("PSNR")
    plt.title('PSNR')
    plt.legend(lines, labels, loc=(1, 0), prop=dict(size=14))
    plt.show()


# Function handle that updates the learning rate
# (note this is a dummy implementation that does nothing)
def base_lr_scheduler(t, T, lr):
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
         batches_per_epoch=None,  # Default: Use entire training set
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
        train_dl = torch.utils.data.DataLoader(
            trainset, batch_size=bs, sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices))
        print("HERRREREREROWAFA")
        # Use one fourth for validation
        subset_indices = list(range(int(np.ceil(batches_per_epoch/4))*bs))
        valid_dl = torch.utils.data.DataLoader(
            testset, batch_size=bs, sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices))

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

    print('Epochs:', epochs, 'Batches per epoch:',
          batches_per_epoch, 'Total number of batches', T)

    # Get initial validation loss and accuracy
    model.eval()
    with torch.no_grad():
        valid_acc = sum(accuracy(model(xb.cuda()), yb.cuda())
                        for xb, yb in valid_dl) / len(valid_dl)
        valid_loss = sum(loss_func(model(xb.cuda()), yb.cuda())
                         for xb, yb in valid_dl) / len(valid_dl)
        valid_loss_history.append(valid_loss.detach().cpu().numpy())
        plot_time_valid.append(t)

    # Train
    for epoch in range(epochs):
        model.train()  # Train mode
        for xb, yb in train_dl:

            # Update learning rate
            opt.param_groups[0]['lr'] = lr_scheduler(
                t, T, lr=opt.param_groups[0]['lr'])

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
                model.eval()  # Test mode
                with torch.no_grad():
                    valid_acc = sum(accuracy(model(xb.cuda()), yb.cuda())
                                    for xb, yb in valid_dl) / len(valid_dl)
                    valid_loss = sum(loss_func(model(xb.cuda()), yb.cuda())
                                     for xb, yb in valid_dl) / len(valid_dl)
                    valid_loss_history.append(
                        valid_loss.detach().cpu().numpy())
                    plot_time_valid.append(t-1)
                    print('t', t, ', lr', opt.param_groups[0]['lr'], 'train loss', loss.detach().cpu().numpy(
                    ), 'val loss', valid_loss.detach().cpu().numpy(), 'val accuracy (PSNR)', valid_acc.detach().cpu().numpy())
                model.train()  # Back to train mode

                # At the end of an epoch:
            if t % bs-1 == 0:  # -1 or not?
                model.eval()  # Test mode
                with torch.no_grad():
                    valid_acc = sum(accuracy(model(xb.cuda()), yb.cuda())
                                    for xb, yb in valid_dl) / len(valid_dl)
                    valid_loss = sum(loss_func(model(xb.cuda()), yb.cuda())
                                     for xb, yb in valid_dl) / len(valid_dl)
                    valid_loss_history.append(
                        valid_loss.detach().cpu().numpy())
                    plot_time_valid.append(t-1)
                    print('Epoch: ', epoch, ', lr', opt.param_groups[0]['lr'], 'train loss', loss.detach().cpu().numpy(
                    ), 'val loss', valid_loss.detach().cpu().numpy(), 'val accuracy', valid_acc.detach().cpu().numpy())
                model.train()  # Back to train mode

    # Summary
    if show_summary:
        plt.figure()
        lines = []
        labels = []
        l, = plt.plot(plot_time_train, train_loss_history)
        lines.append(l)
        labels.append('Training')
        l, = plt.plot(plot_time_valid, valid_loss_history)
        lines.append(l)
        labels.append('Validation')
        plt.title('Loss')
        plt.legend(lines, labels, loc=(1, 0), prop=dict(size=14))
        plt.show()

    return train_loss_history


def MSE(input, target):
    # Crop the target to the input size - the input loses some of it's size as the model doesn't use any padding
    crop_transform = transforms.CenterCrop(
        size=(input.size(dim=2), input.size(dim=3)))
    tgt = crop_transform(target)

    #print("Left top corner of target:" + str(target[0,1,6,6]))
    #print("Right bottom corner of target:" + str(target[0,1,26,26]))

    #print("Left top corner of tgt:" + str(tgt[0,1,0,0]))
    #print("Right bottom corner of tgt:" + str(tgt[0,1,20,20]))
    # show_tensor_as_image(input[None,0])
    #show_tensor_as_image(tgt[None, 0])
    # loss = 0
    # bs = input.size(dim = 0)
    # for i in range(bs):
    #     loss += MSE_Single(input[i], tgt[i])

    # return loss / bs

    mse = F.mse_loss
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
        nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False),

        # Model
        nn.Conv2d(channels, n1, (f1, f1)),
        nn.ReLU(),
        nn.Conv2d(n1, n2, (f2, f2)),
        nn.ReLU(),
        nn.Conv2d(n2, channels, (f3, f3)),
        # nn.ReLU() #this should not be here - it is the last layer!
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

        self.params = nn.ModuleDict({
            'conv12': nn.ModuleList([self.conv1, self.conv2]),
            'conv3': nn.ModuleList([self.conv3])})

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        #x = self.relu3(x)
        return x


class ModularFSRCNN(nn.Module):
    def __init__(self):
        d = 56
        s = 12
        m = 4
        scale = 3

        super(ModularFSRCNN, self).__init__()
        #nn.Conv2d(input_channels, output_channels, kernel_size)
        # Feature extraction
        self.conv1 = nn.Conv2d(3, d, (5, 5), padding=2)
        self.relu1 = nn.PReLU()

        # Shrinking
        self.conv2 = nn.Conv2d(d, s, (1, 1))
        self.relu2 = nn.PReLU()

        # Mapping m=4 hardcoded for now TODO: parameterize it
        self.convm1 = nn.Conv2d(s, s, (3, 3), padding=1)
        self.relum1 = nn.PReLU()
        self.convm2 = nn.Conv2d(s, s, (3, 3), padding=1)
        self.relum2 = nn.PReLU()
        self.convm3 = nn.Conv2d(s, s, (3, 3), padding=1)
        self.relum3 = nn.PReLU()
        self.convm4 = nn.Conv2d(s, s, (3, 3), padding=1)
        self.relum4 = nn.PReLU()

        # Expansion
        self.conv3 = nn.Conv2d(s, d, (1, 1))
        self.relu3 = nn.PReLU()

        # Deconvolution
        self.deconv = nn.ConvTranspose2d(d, 3, (9, 9), stride=scale)

        self.params = nn.ModuleDict({
            'convs': nn.ModuleList([self.conv1, self.conv2, self.convm1, self.convm2, self.convm3, self.convm4, self.conv3]),
            'deconvs': nn.ModuleList([self.deconv])})

    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.relu1(x)

        # Shrinking
        x = self.conv2(x)
        x = self.relu2(x)

        # Mapping
        x = self.convm1(x)
        x = self.relum1(x)
        x = self.convm2(x)
        x = self.relum2(x)
        x = self.convm3(x)
        x = self.relum3(x)
        x = self.convm4(x)
        x = self.relum4(x)

        # Expansion
        x = self.conv3(x)
        x = self.relu3(x)

        # Deconvolution
        x = self.deconv(x)

        return x


def setupSRCNN(load_model=False, load_path="./Models/SRCNN.pth", lr=0.001):
    model = ModularSRCNN().cuda()

    if load_model:
        model.load_state_dict(torch.load(load_path))
    else:
        reset_parameters(model)

    loss_func = MSE
    optim = adam_SGD(model, lr=lr)

    return model, loss_func, optim


def loadSRCNNdata(bs=128, n=-1, scale=3):
    if(scale == 3):
        train_dl = get_preprocessed_dataloader(
            "./Datasets/T91/T91_Upscaled_Patches", "./Datasets/T91/T91_HR_Patches", n=n, bs=bs)
        test_dl = get_preprocessed_dataloader(
            "./Datasets/Set19/Blurred", "./Datasets/Set19/Original", bs=1)
    elif(scale == 2):
        train_dl = get_preprocessed_dataloader(
            "./Datasets/T91/T91_Upscaled_Patches_x2", "./Datasets/T91/T91_HR_Patches_x2", n=n, bs=bs)
        test_dl = get_preprocessed_dataloader(
            "./Datasets/Set19/Blurred_x2", "./Datasets/Set19/Original", bs=1)
    return train_dl, test_dl


def setupFSRCNN(load_model=False, load_path="./Models/FSRCNN.pth", lr=0.001):
    model = ModularFSRCNN().cuda()

    if load_model:
        model.load_state_dict(torch.load(load_path))
    else:
        reset_parameters(model)

    loss_func = MSE
    optim = adam_SGD_faster(model, lr=lr)

    return model, loss_func, optim


def loadFSRCNNdata(bs=128, n=-1, scale=3):
    if(scale == 3):
        train_dl = get_preprocessed_dataloader(
            "./Datasets/T91/T91_LR_Patches", "./Datasets/T91/T91_HR_Patches", n=n, bs=bs)
        test_dl = get_preprocessed_dataloader(
            "./Datasets/Set19/LR", "./Datasets/Set19/Original", bs=1)
    elif(scale == 2):
        train_dl = get_preprocessed_dataloader(
            "./Datasets/T91/T91_LR_Patches_x2", "./Datasets/T91/T91_HR_Patches_x2", n=n, bs=bs)
        test_dl = get_preprocessed_dataloader(
            "./Datasets/Set19/LR_x2", "./Datasets/Set19/Original", bs=1)
    return train_dl, test_dl


class BicubicBaseline(nn.Module):
    def __init__(self):
        super(BicubicBaseline, self).__init__()

    def forward(self, x):
        return x

# Function handle that returns an optimizer
# We have learning rate 0.0001 for the first two layers and 0.00001 for the last layer


def basic_SGD(model, lr=0.0001, momentum=0.9):
    return optim.SGD([
        {'params': model.params.conv12.parameters()},
        {'params': model.params.conv3.parameters(), 'lr': 0.00001}
    ], lr=lr, momentum=momentum)


def adam_SGD(model, lr=0.0001):
    return optim.Adam([
        {'params': model.params.conv12.parameters()},
        {'params': model.params.conv3.parameters(), 'lr': lr*0.1}
    ], lr=lr)


def adam_SGD_faster(model, lr=0.001):
    return optim.Adam([
        {'params': model.params.convs.parameters()},
        {'params': model.params.deconvs.parameters(), 'lr': lr*0.1}
    ], lr=lr)


def reset_parameters(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            print("init'ing conv2d layer...")
            #torch.nn.init.normal_(m.weight, mean=0.0, std=0.001)
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.ConvTranspose2d):
            print("conv2d transpose parameter reset")
            nn.init.normal_(m.weight, mean=0, std=0.001)
        '''elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1) # Why 1?
            torch.nn.init.constant_(m.bias, 0) # Why 0?
        elif isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)'''


def get_preprocessed_dataloaders(lowPath, highPath, n=-1, bs=8):
    train_dataset, test_dataset, val_dataset = get_preprocessed_image_sets(
        lowPath, highPath, n)
    train_dl = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=bs, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=bs, shuffle=True)
    return train_dl, test_dl, val_dl


def show_tensor_as_image(tensor, title=None):
    if tensor.ndimension() == 4:
        tensor = tensor[0]

    tensor_pil_converter = transforms.ToPILImage()
    tensor_pil_converter(tensor).show(title=title)


def PSNRaccuracy(scores, yb, max_val=1):
    crop_transform = transforms.CenterCrop(
        size=(scores.size(dim=2), scores.size(dim=3)))
    yb_cropped = crop_transform(yb)

    diff = scores - yb_cropped
    diff_flat = torch.flatten(diff, start_dim=1)

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
    # return (preds == yb).float().mean()


# This function works on preprocessed data
def get_preprocessed_image_sets(path_low, path_high, n=-1):
    dataset = get_tensor_images_high_low(path_low, path_high, n)
    n = len(dataset)
    # Calculate the sizes of the data subsets
    # We need to check the size difference to correct for rounding errors
    trainSize = n*8//10
    testSize = n//10
    valSize = n//10
    sizeDiff = trainSize + testSize + valSize - n
    trainSize -= sizeDiff  # correct for rounding errors
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [trainSize, testSize, valSize])
    return train_dataset, test_dataset, val_dataset


def get_preprocessed_image_set(path_low, path_high, n=-1):
    return get_tensor_images_high_low(path_low, path_high, n)


def get_preprocessed_dataloader(lowPath, highPath, n=-1, bs=8):
    dataset = get_preprocessed_image_set(lowPath, highPath, n)
    train_dl = DataLoader(dataset, batch_size=bs, shuffle=True)
    return train_dl


# This function works on preprocessed data
def get_tensor_images_high_low(path_low, path_high, n=-1):
    images_low = []
    images_high = []
    pil_tensor_converter = transforms.ToTensor()
    # Convert all images on the given path to tensors
    counter_low = n
    for f in sorted(glob.iglob(path_low + "/*")):
        # Break if n images appended (if n = -1 - aka get all images - this never occurs)
        if counter_low == 0:
            break
        im = Image.open(f)
        img = pil_tensor_converter(im)
        images_low.append(img)
        counter_low -= 1

        #print("Image mode: " + str(im.mode))
        #print("The values of image:" + str(img))

    print("Done with lowres...")
    counter_high = n
    for f in sorted(glob.iglob(path_high + "/*")):
        # Break if n images appended (if n = -1 - aka get all images - this never occurs)
        if counter_high == 0:
            break
        im = Image.open(f)
        # print("Image mode: " + str(im.mode)) - The images are RGB
        img = pil_tensor_converter(im)
        images_high.append(img)

        counter_high -= 1
    print("Done with highres...")
    # low_res_stack = torch.stack(images_low)
    # high_res_stack = torch.stack(images_high)

    # return TensorDataset(low_res_stack, high_res_stack)
    return ImageDataset(images_low, images_high)


def get_random_images_for_prediction(n=1, scale=3):
    # path = "./Datasets/T91/T91_Original/*"
    path = "./Datasets/Set19/Original/*"
    files = [f for f in glob.iglob(path)]
    random.shuffle(files)

    upscaled_imgs = []
    original_imgs = []
    low_res_imgs = []

    tensor_transform = transforms.ToTensor()

    counter = 0

    for f in files:

        if counter >= n:
            break
        pic = Image.open(f)
        img = np.array(pic)

        h, w, _ = img.shape
        low_res_img = cv2.resize(
            img, (int(w*1/scale), int(h*1/scale)), interpolation=cv2.INTER_CUBIC)
        
        high_res_upscale = cv2.resize(low_res_img, (w, h),
                                        interpolation=cv2.INTER_CUBIC)

        low_res_imgs.append(tensor_transform(low_res_img))
        upscaled_imgs.append(tensor_transform(high_res_upscale))
        original_imgs.append(tensor_transform(img))

        counter += 1

    # img_dl = TensorDataset(torch.stack(upscaled_imgs), torch.stack(original_imgs))
    lr_img_ds = ImageDataset(low_res_imgs,original_imgs)
    upscaled_img_ds = ImageDataset(upscaled_imgs, original_imgs)
    lr_dataloader = DataLoader(lr_img_ds, batch_size=1, shuffle=False)
    upscaled_dataloader = DataLoader(upscaled_img_ds, batch_size=1, shuffle=False)
    return lr_dataloader, upscaled_dataloader

# ------------------- CODE -------------------


def main():
    bs = 128
    epochs = 200

    # train_dl, test_dl = loadFSRCNNdata(bs=bs)
    modelFSRCNN, loss_func, optim = setupFSRCNN(load_model = True, lr=0.001)
    #fit(epochs, modelFSRCNN, loss_func, optim, train_dl, test_dl)

    # # # Save the model
    # print("Saving model after training")
    # torch.save(modelFSRCNN.state_dict(), "./Models/FSRCNN.pth")

    lr_dl, upscaled_dl = get_random_images_for_prediction(scale=3)

    modelFSRCNN.eval()
    for xb, yb in lr_dl:
        input = xb.cuda()[None, 0]
        result = modelFSRCNN(input).clamp(0, 1)

        show_tensor_as_image(result, "result")
        show_tensor_as_image(yb[0], "original")
        show_tensor_as_image(input, "input")
    
    for xb, yb in upscaled_dl:
        blurred = xb.cuda()[None, 0]

        show_tensor_as_image(blurred, "blurred")

    # fit2(model, training, validation, loss_func, PSNRaccuracy, basic_SGD, bs=bs, epochs=epochs)

    # counter = 4
    # for xb, yb in train_dl:
    #     if(counter == 0):
    #         break
    #     print(str(xb.cuda()[None,0].size()))
    #     model.eval()
    #     result = model(xb.cuda()[None, 0])
    #     show_tensor_as_image(result)
    #     print("result size: " + str(result.size()))
    #     crop_transform = transforms.CenterCrop(result.size(dim=2))
    #     tgt = crop_transform(yb.cuda()[0])
    #     print("tgt size: " + str(tgt.size()))

    #     show_tensor_as_image(tgt)
    #     counter -= 1
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
