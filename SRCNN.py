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
from BigNet import BigNet



# https://github.com/yjn870/SRCNN-pytorch/blob/master/train.py
# https://medium.com/coinmonks/review-srcnn-super-resolution-3cb3a4f67a7c

#
angles = [0, 90, 180, 270]

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

def loss_batch(model, loss_func, xb, yb, opt=None, data_augmentation=False):
    #print("xb max before:", torch.max(xb))
    if data_augmentation:
        xb,yb = apply_image_transformations(xb,yb)

    #print("xb max fater:", torch.max(xb))

    predictions_unclamped = model(xb.cuda())
    loss = loss_func(predictions_unclamped, yb.cuda())
    psnr = PSNRaccuracy(predictions_unclamped.clamp(0, 1), yb.cuda())

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb), psnr

#Identity lr_scheduler for fit2

#Identity lr_scheduler for fit

class base_lr_schedule():
    def __init__(self):
        pass

    def update_learning_rate(self, curr_epoch, curr_batch, epoch_batches, num_epochs, lr):
        return lr


#Tests learning rates between 10e-6 and 10e-1 such that the increments are even on a logarithmic scale on the learning rate
def opt_lr_finding_schedule(curr_epoch, curr_batch, epoch_batches, num_epochs, lr):
    #print("lr:", lr)
    #min_lr for bignet:0.0000001, max_lr: 1
    #min_lr for rest: 0.000001, max_lr: 0.1
    min_lr = 0.0000001
    max_lr = 0.1
    step_size = abs(math.log(max_lr/min_lr) / (epoch_batches*num_epochs))
    if(lr > max_lr):
        return lr + min_lr
    lr = math.e**(math.log(lr)+step_size*2)
    return lr



class cyclic_lr_schedule:
    def __init__(self, max_lr, min_lr, cycle_length=20):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_length = cycle_length

    def update_learning_rate(self, curr_epoch, curr_batch, epoch_batches, num_epochs, lr):
        cycle_length_epochs = self.cycle_length
        if curr_epoch % cycle_length_epochs == 0 and curr_epoch != 0:
            if self.max_lr > 2* self.min_lr:
                print("Updating max...")
                self.max_lr = self.max_lr * 0.5
        step_size = (self.max_lr - self.min_lr) / (cycle_length_epochs / 2)
        mod_val = curr_epoch % cycle_length_epochs
        if mod_val < 10:
            print("step_size:", step_size, "mod val:", mod_val, "curr_epoch", curr_epoch, "cyclelengthepochs", cycle_length_epochs)
            lr = self.min_lr + mod_val * step_size
            print("lr", lr)
        else:
            print("step_size:", step_size, "mod val:", mod_val)
            lr = self.max_lr - (mod_val - 10) * step_size
            print("lr", lr)
        return lr




def fit(model, loss_func, opt, train_dl, valid_dl, epochs, save_path="FSCRNN", load_model=False, lr_scheduler=None, data_augmentation=False):
    num_batches = len(train_dl)
    # Total number of batches
    num_elements = len(train_dl.dataset)
    load_state =  "preloaded" if load_model else "new" 
    print("Running Fit with", load_state, "model:", str(type(model).__name__), "and data augmentation set to", data_augmentation)
    print('Epochs:', epochs, 'Total number of batches', num_batches, 'Total number of elements', num_elements)

    val_psnr_history = []
    training_psnr_history = []
    val_loss_history = []
    training_loss_history = []
    t_counter = 0
    t = []
    best_psnr = 0
    best_weights = model.state_dict()        

    if(load_model):
        try:
            best_psnr = np.load("./Models/" + save_path + "_best_psnr.npy", allow_pickle=True)[0]
        except:
            best_psnr = 0
        t = np.load("./Models/" + save_path + "_T_val.npy", allow_pickle=True).tolist()
        t_counter = t[len(t)-1]
        val_psnr_history = np.load("./Models/" + save_path + "_val_psnr.npy", allow_pickle=True).tolist()
        training_psnr_history = np.load("./Models/" + save_path + "_training_psnr.npy", allow_pickle=True).tolist()
        val_loss_history = np.load("./Models/" + save_path + "_val_loss.npy", allow_pickle=True).tolist()
        training_loss_history = np.load("./Models/" + save_path + "_training_loss.npy", allow_pickle=True).tolist()

    for epoch in range(epochs):
        model.train()  # Sets the mode of the model to training

        if lr_scheduler is not None:
            # Update learning rate
            for param_group in opt.param_groups:
                new_lr = lr_scheduler.update_learning_rate(
                    t_counter, 0, num_batches, num_epochs=epochs, lr=param_group['lr'])
                print("New lr:", new_lr)
                param_group['lr'] = new_lr


        # for xb, yb in train_dl:
        #    loss_batch(model, loss_func, xb, yb, opt)
        loss_losses, loss_nums, loss_psnrs = zip(
            *[loss_batch(model, loss_func, xb, yb, opt, data_augmentation=data_augmentation) for xb, yb in train_dl]
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

        training_psnr_history.append(train_psnr)
        val_psnr_history.append(val_psnr)
        training_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        t_counter += 1
        t.append(t_counter)

        # print("Epoch: " + str(epoch) + " Val loss: " + str(val_loss) + " Val PSNR: " + str(val_psnr))
        # print("Epoch: " + str(epoch) + " Train loss: " + str(train_loss) + " Val loss: " + str(val_loss) + " Val PSNR: " + str(val_psnr))
        print("Epoch: " + str(t_counter) + " Train loss: " + str(train_loss) + " Train PSNR: " +
              str(train_psnr) + " Val loss: " + str(val_loss) + " Val PSNR: " + str(val_psnr))

        if val_psnr > best_psnr:
            best_weights = model.state_dict()
            best_psnr = val_psnr

        if (epoch + 1) % 25 == 0:
            # Save the model
            print("Saving model weights...")
            # torch.save(model.state_dict(), "./Models/" + save_path + ".pth")
            torch.save(best_weights, "./Models/" + save_path + ".pth")

            print("saved best psnr: " + str(best_psnr))

            #Save the graphed results:
            print("Saving graphs...")
            np.save("./Models/" + save_path + "_best_psnr.npy", best_psnr)
            np.save("./Models/" + save_path + "_T_val.npy", t)
            np.save("./Models/" + save_path + "_val_psnr.npy", val_psnr_history)
            np.save("./Models/" + save_path + "_training_psnr.npy", training_psnr_history)
            np.save("./Models/" + save_path + "_val_loss.npy", val_loss_history)
            np.save("./Models/" + save_path + "_training_loss.npy", training_loss_history)


    plt.figure()
    lines = []
    labels = []
    # l, = plt.plot(plot_time_train,train_loss_history)
    # lines.append(l)
    # labels.append('Training')
    # l, = plt.plot(plot_time_valid,valid_loss_history)
    # lines.append(l)
    # labels.append('Validation')
    l, = plt.plot(t, val_psnr_history)
    lines.append(l)
    labels.append("Validation PSNR")
    l, = plt.plot(t, training_psnr_history)
    lines.append(l)
    labels.append("Training PSNR")
    plt.grid()
    plt.title('PSNR')
    plt.legend(lines, labels, loc=(1, 0), prop=dict(size=14))
    plt.show()



def testModel(model, loss_func, testset_dl):
    model.eval()

    with torch.no_grad():
            losses, nums, test_psnrs = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in testset_dl]
            )
    test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
    test_psnr = np.sum(test_psnrs) / len(test_psnrs)  # average PSNR per batch

    print("Test loss: " + str(test_loss) + " Test PSNR: " +  str(test_psnr))
    



def PSNRaccuracy(scores, yb, max_val=1):
    # crop_transform = transforms.CenterCrop(
    #     size=(scores.size(dim=2), scores.size(dim=3)))
    # yb_cropped = crop_transform(yb)
    #print("Scores size", scores.size())
    #print("yb size", yb.size())
    diff = scores - yb
    diff_flat = torch.flatten(diff, start_dim=1)
    #print("Max value: ", str(yb.max()))
    rmse = torch.sqrt(torch.mean(torch.square(diff_flat)))

    return Tensor([20 * math.log(max_val/rmse, 10)])


#This is used to find optimal learning rate ranges since we can work on a per-batch basis
def fit2(model,
         loss_func,
         opt,
         trainset,
         testset,
         epochs=1,
         accuracy=PSNRaccuracy,
         batch_prints=False,
         lr_scheduler=base_lr_schedule,
         batches_per_epoch=None,  # Default: Use entire training set
         show_summary=True,
         data_aug = False):

    bs = trainset.batch_size
    num_batches = len(trainset)
    learning_rate = []

    # Set up data loaders
    if batches_per_epoch == None:
        # Use all images
        train_dl = trainset
        #torch.utils.data.DataLoader(trainset, batch_size=bs,  shuffle=True, num_workers=2)
        valid_dl = testset
        #torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)
        batches_per_epoch = len(train_dl)
    else:
        print("Yo im here")
        # Only use a subset of the data
        subset_indices = list(range(batches_per_epoch*bs))
        dataset = trainset.dataset        
        train_dl = DataLoader(dataset, batch_size=bs, sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices), shuffle=True)
        valid_dl = testset
        num_batches = len(train_dl)


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
            if lr_scheduler is not None:
            # Update learning rate
                for param_group in opt.param_groups:
                    new_lr = lr_scheduler(
                        epoch, t, batches_per_epoch, epochs, lr=param_group['lr'])
                    param_group['lr'] = new_lr
                learning_rate.append(opt.param_groups[0]['lr'])
            
            if data_aug:
                xb,yb = apply_image_transformations(xb,yb)

            
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
            if t % num_batches == 0:
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

    if lr_scheduler == opt_lr_finding_schedule:
        plt.figure()
        lines = []
        labels = []
        l, = plt.plot(learning_rate, train_loss_history)
        lines.append(l)
        labels.append("Loss")
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.ticklabel_format(axis='x', style='scientific')
        plt.xscale('log')
        #plt.ylim([0, 7000]) for bignet
        plt.ylim([0, 1])
        plt.title('Learning rate analysis')
        plt.legend(lines, labels, loc=(1, 0), prop=dict(size=14))
        plt.show()

    return train_loss_history

def MSE(input, target):
    mse = F.mse_loss
    return mse(input, target)

#Computes the absolute difference between the pixel values (instead of the squared difference)
def MAE(input, target):
    mae = F.l1_loss
    return mae(input, target)

class ModularSRCNN(nn.Module):

    # This model has 8129 parameters when channels = 1 (20099 when channels = 3) including bias, 8032 without - which is as in the article.

    def __init__(self):
        channels = 3
        n1 = 64
        n2 = 32
        f1 = 9
        f2 = 1
        f3 = 5

        super(ModularSRCNN, self).__init__()

        self.conv1 = nn.Conv2d(channels, n1, (f1, f1), padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(n1, n2, (f2, f2))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(n2, channels, (f3, f3), padding=2)
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
        #ConvTranspose2d(in_channels, out_channels, kernel_size, padding, stride)
        #Takes input of size nxn. From this, it takes the (k x k) kernel and uses it for each value in the nxn input
        #If stride is 3, with kernel size (9,9), it means that we make a new matrix where we start in the top corner
        #making a 9x9 matrix. Then we slide by 3 to the right, and we now have a 12x9 matrix. This means we get 9+(n-1)x3 size in total (we don't slide for the first of the n values)
        #If we add padding=3 it means we treat a 3x3 border around the result as not part of the result. It makes sense since 3 results on either side will not be an overlay of three inputs with the filter.  
        self.deconv = nn.ConvTranspose2d(d, 3, (9, 9), padding=scale, stride=scale)


        self.params = nn.ModuleDict({
           'convs': nn.ModuleList([self.conv1, self.conv2, self.convm1, self.convm2, self.convm3, self.convm4, self.conv3
           , self.relum1, self.relum2, self.relum3, self.relum4, self.relu1, self.relu2, self.relu3
           ]),
           'deconvs': nn.ModuleList([self.deconv])})
        # self.params = nn.ModuleDict({
        #    'convs': nn.ModuleList([]),
        #    'deconvs': nn.ModuleList([])})

    def forward(self, x):
        #print("x init size: " + str(x.size()))
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
        #print("x after deconv size: " + str(x.size()))

        return x






class BRFSRCNN(nn.Module):
    def __init__(self):
        d = 64
        intermediate = 32
        s1 = 16
        m = 4
        scale = 3

        super(BRFSRCNN, self).__init__()
        #nn.Conv2d(input_channels, output_channels, kernel_size)
        self.relu = nn.ReLU()
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()


        # Feature extraction
        self.conv1 = nn.Conv2d(3, 64, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(64, 64, (5, 5), padding=2)
        self.conv3 = nn.Conv2d(64, 64, (5, 5), padding=2)
        self.bnorm1 = nn.BatchNorm2d(num_features=64)

        # Shrinking
        self.conv4 = nn.Conv2d(64, 32, (1, 1))
        self.conv5 = nn.Conv2d(32, 16, (1, 1))

        # Mapping
        self.convm1 = nn.Conv2d(16, 16, (3, 3), padding=1)
        self.bnorm2  = nn.BatchNorm2d(16)
        self.convm2 = nn.Conv2d(16, 16, (3, 3), padding=1)
        self.bnorm3  = nn.BatchNorm2d(16)
        
        self.convm3 = nn.Conv2d(16, 16, (3, 3), padding=1)
        self.bnorm4  = nn.BatchNorm2d(16)
        self.convm4 = nn.Conv2d(16, 16, (3, 3), padding=1)
        self.bnorm5  = nn.BatchNorm2d(16)
        
        self.convm5 = nn.Conv2d(16, 16, (3, 3), padding=1)
        self.bnorm6  = nn.BatchNorm2d(16)
        self.convm6 = nn.Conv2d(16, 16, (3, 3), padding=1)
        self.bnorm7  = nn.BatchNorm2d(16)
        
        self.convm7 = nn.Conv2d(16, 16, (3, 3), padding=1)
        self.bnorm8  = nn.BatchNorm2d(16)
        self.convm8 = nn.Conv2d(16, 16, (3, 3), padding=1)
        self.bnorm9  = nn.BatchNorm2d(16)

        # Expansion
        self.conv6 = nn.Conv2d(s1, intermediate, (1, 1))
        self.conv7 = nn.Conv2d(intermediate, d, (1, 1))

        # Deconvolution
        #ConvTranspose2d(in_channels, out_channels, kernel_size, padding, stride)
        #Takes input of size nxn. From this, it takes the (k x k) kernel and uses it for each value in the nxn input
        #If stride is 3, with kernel size (9,9), it means that we make a new matrix where we start in the top corner
        #making a 9x9 matrix. Then we slide by 3 to the right, and we now have a 12x9 matrix. This means we get 9+(n-1)x3 size in total (we don't slide for the first of the n values)
        #If we add padding=3 it means we treat a 3x3 border around the result as not part of the result. It makes sense since 3 results on either side will not be an overlay of three inputs with the filter.  
        self.deconv = nn.ConvTranspose2d(d, 3, (9, 9), padding=scale, stride=scale)

        self.params = nn.ModuleDict({
            'convs': nn.ModuleList([self.conv1, self.conv2, self.conv3, self.bnorm1, self.conv4, self.conv5, self.convm1, self.bnorm2, self.convm2, self.bnorm3, self.convm3, self.bnorm4, self.convm4, self.bnorm5, self.convm5, self.bnorm6, self.convm6, self.bnorm7, self.convm7, self.bnorm8, self.convm8, self.bnorm9, self.conv6, self.conv7]),
           'deconvs': nn.ModuleList([self.deconv])})
        # self.params = nn.ModuleDict({
        #    'convs': nn.ModuleList([]),
        #    'deconvs': nn.ModuleList([])})

    def forward(self, x):
        #print("x init size: " + str(x.size()))
        # Feature extraction
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bnorm1(x)
        x = self.relu(x)

        # Shrinking
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)

        # Mapping
        identity = x
        x = self.convm1(x)
        x = self.bnorm2(x)
        x = self.relu(x)
        x = self.convm2(x)
        x = self.bnorm3(x)
        x = torch.add(x, identity)

        identity = x
        x = self.convm3(x)
        x = self.bnorm4(x)
        x = self.relu(x)
        x = self.convm4(x)
        x = self.bnorm5(x)
        x = torch.add(x, identity)

        identity = x
        x = self.convm5(x)
        x = self.bnorm6(x)
        x = self.relu(x)
        x = self.convm6(x)
        x = self.bnorm7(x)
        x = torch.add(x, identity)

        identity = x
        x = self.convm7(x)
        x = self.bnorm8(x)
        x = self.relu(x)
        x = self.convm8(x)
        x = self.bnorm9(x)
        x = torch.add(x, identity)

        # Expansion
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.relu(x)

        # Deconvolution
        x = self.deconv(x)
        #print("x after deconv size: " + str(x.size()))

        return x




def setupSRCNN(load_model=False, load_path="SRCNN", lr=0.001):
    model = ModularSRCNN().cuda()

    if load_model:
        model.load_state_dict(torch.load("./Models/"+ load_path + ".pth" ))
    else:
        reset_parameters(model)

    loss_func = MSE
    optim = adam_SGD(model, lr=lr)

    return model, loss_func, optim, load_path


def loadSRCNNdata(bs=128, n=-1, scale=3):
    if(scale == 3):
        train_dl = get_preprocessed_dataloader(
            "./Datasets/T91/T91_Upscaled_Patches_x3gaussPIL", "./Datasets/T91/T91_HR_Patches", n=n, bs=bs)
        test_dl = get_preprocessed_dataloader(
            "./Datasets/Set19/Upscaled_x3gaussPIL", "./Datasets/Set19/OriginalCropx3", bs=1)
    elif(scale == 2):
        train_dl = get_preprocessed_dataloader(
            "./Datasets/T91/T91_Upscaled_Patches_x2", "./Datasets/T91/T91_HR_Patches_x2", n=n, bs=bs)
        test_dl = get_preprocessed_dataloader(
            "./Datasets/Set19/Blurred_x2", "./Datasets/Set19/Original", bs=1)
    return train_dl, test_dl


def setupFSRCNN(load_model=False, load_path="FSRCNN", lr=0.001):
    model = ModularFSRCNN().cuda()

    if load_model:
        model.load_state_dict(torch.load("./Models/"+ load_path + ".pth" ))
    else:
        reset_parameters(model)

    loss_func = MSE
    optim = adam_SGD_faster(model, lr=lr)

    return model, loss_func, optim, load_path

def setupBRFSRCNN(load_model= False, load_path="BRFSRCNN", lr=0.001):
    model = BRFSRCNN().cuda()

    if load_model:
        model.load_state_dict(torch.load("./Models/"+ load_path + ".pth" ))
    else:
        reset_parameters(model)

    loss_func = MSE
    optim = adam_SGD_DeepNet(model, lr=lr)

    return model, loss_func, optim, load_path

def setupBigNet(load_model=False, load_path="BigNet", lr=0.0001):
    model = BigNet().cuda()

    if load_model:
        model.load_state_dict(torch.load("./Models/" + load_path + ".pth"))
    else:
        reset_parameters(model)

    loss_func = MAE
    optim = adam_SGD_BigNet(model, lr=lr)

    return model, loss_func, optim, load_path

def loadFSRCNNdata(bs=128, n=-1, scale=3):
    if(scale == 3):
        train_dl = get_preprocessed_dataloader(
            "./Datasets/T91/T91_LR_Patches_x3gaussPIL", "./Datasets/T91/T91_HR_Patches", n=n, bs=bs)
        test_dl = get_preprocessed_dataloader(
            "./Datasets/Set19/LR_x3gaussPIL", "./Datasets/Set19/OriginalCropx3", bs=1, shuffle=False)
    elif(scale == 2):
        train_dl = get_preprocessed_dataloader(
            "./Datasets/T91/T91_LR_Patches_x2", "./Datasets/T91/T91_HR_Patches_x2", n=n, bs=bs)
        test_dl = get_preprocessed_dataloader(
            "./Datasets/Set19/LR_x2", "./Datasets/Set19/Original", bs=1, shuffle=False)
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

def adam_SGD_DeepNet(model, lr=0.001):
    return optim.Adam([
        {'params': model.params.convs.parameters()},
        {'params': model.params.deconvs.parameters(), 'lr': lr*0.1}
    ], lr=lr)


def adam_SGD_BigNet(model, lr=0.001):
    return optim.Adam(model.parameters(), lr=lr)

def reset_parameters(net):
    '''Init layer parameters.'''
    counter = 0
    print("Resetting parameters...")
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            #torch.nn.init.normal_(m.weight, mean=0, std=0.001)
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)            
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, mean=0, std=0.001)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1) # Why 1?
            torch.nn.init.constant_(m.bias, 0) # Why 0?
        elif isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        #print("Reseting layer: " + str(counter))
        counter += 1
    

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
    return get_tensor_images_high_low(path_low, path_high)


def get_preprocessed_dataloader(lowPath, highPath, n=-1, bs=8, shuffle=True):
    dataset = get_preprocessed_image_set(lowPath, highPath, n)
    train_dl = DataLoader(dataset, batch_size=bs, shuffle=shuffle)
    return train_dl


# This function works on preprocessed data
def get_tensor_images_high_low(path_low, path_high, n=-1):
    start_time = time.time()
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
        #print("Low name: ", (f.split("\\")[1]).split(".")[0])
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
        #print("High name: ", (f.split("\\")[1]).split(".")[0])
        counter_high -= 1
    print("Done with highres...")
    # low_res_stack = torch.stack(images_low)
    # high_res_stack = torch.stack(images_high)

    # return TensorDataset(low_res_stack, high_res_stack)
    # return ImageDataset(images_low, images_high)

    print("Time to get dataset: " + str(time.time() - start_time))
    return ImageDataset(images_low, images_high)

def get_random_images_for_prediction(n=1, scale=3, listOfImages=[]):
    path = "./Datasets/Set19/OriginalCropx3/*"
    files = [f for f in glob.iglob(path)]
    random.shuffle(files)

    upscaled_imgs = []
    original_imgs = []
    low_res_imgs = []
    picture_names = []
    tensor_transform = transforms.ToTensor() #Converts from 0,255 to 0,1
    gaussian_kernel = transforms.GaussianBlur(kernel_size=(3,3), sigma=0.7)

    counter = 0
    for f in files:
        if len(listOfImages) ==  0:
            if counter >= n:
                break
        else:
            name = (f.split("\\")[1]).split(".")[0]
            if name not in listOfImages:
                #print("This is not in the list...")
                continue
            #print("Whew, ", str(name), "is in the list", str(listOfImages)) 
            
            
        picture_names.append(name)
        pic = Image.open(f)
        img = np.array(pic)
    
        h, w, _ = img.shape
        # low_res_img = cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
        # print("Low_res_img max: ", low_res_img.max())
        # high_res_upscale = cv2.resize(low_res_img, (w, h), interpolation=cv2.INTER_CUBIC)
        
        gaus_pic = gaussian_kernel(pic)
        low_res = gaus_pic.resize(size=(w//scale,h//scale), resample=Image.BICUBIC)
        upscaled = low_res.resize(size=(w,h), resample=Image.BICUBIC)
        low_res_img = np.array(low_res)
        high_res_upscale = np.array(upscaled)

        low_res_imgs.append(tensor_transform(low_res_img))
        upscaled_imgs.append(tensor_transform(high_res_upscale))
        original_imgs.append(tensor_transform(img))

        counter += 1

    # img_dl = TensorDataset(torch.stack(upscaled_imgs), torch.stack(original_imgs))
    lr_img_ds = ImageDataset(low_res_imgs,original_imgs)
    upscaled_img_ds = ImageDataset(upscaled_imgs, original_imgs)
    lr_dataloader = DataLoader(lr_img_ds, batch_size=1, shuffle=False)
    upscaled_dataloader = DataLoader(upscaled_img_ds, batch_size=1, shuffle=False)
    return lr_dataloader, upscaled_dataloader, picture_names

angles = [0, 90, 180, 270]
def apply_image_transformations(input, target):
    horizontalFlip = transforms.RandomHorizontalFlip(p=1)
    greyscale = transforms.RandomGrayscale(p=1)
    invert = transforms.RandomInvert(p=1)

    flip = random.randint(0,1)
    #greyScale = random.randint(0,1)
    #inversion = random.randint(0,1)

    transformedInput = torch.clone(input)
    transformedTarget = torch.clone(target)
    if flip:
      transformedInput = horizontalFlip(transformedInput)
      transformedTarget = horizontalFlip(transformedTarget)

    #if greyScale:
    #  transformedInput = greyscale(transformedInput)
    #  transformedTarget = greyscale(transformedTarget)

    # if inversion:
    #   transformedInput = invert(transformedInput)
    #   transformedTarget = invert(transformedTarget)
      


    # Rotatus maximus
    degrees = random.choice(angles)
    transformedInput = transforms.functional.rotate(transformedInput, degrees)
    transformedTarget = transforms.functional.rotate(transformedTarget, degrees)

    return transformedInput, transformedTarget


def evaluateModel(model, lr_dl, upscaled_dl, picture_numbers, show_images=True):
    model.eval()

    counter1 = 0
    for xb, yb in lr_dl:
        input = xb.cuda()[None, 0]
        result = model(input)
        clampedResult = result.clamp(0, 1)

        print(str(picture_numbers[counter1]) + " upscaled PSNR: " + str(PSNRaccuracy(result, yb[0].cuda())))
        if show_images:
            show_tensor_as_image(clampedResult, "result")
            show_tensor_as_image(yb[0], "original")
            show_tensor_as_image(input, "input")
        counter1 += 1
        

    counter2 = 0
    for xb, yb in upscaled_dl:
        blurred = xb.cuda()[None, 0]
        print(str(picture_numbers[counter2]) + " bicubic upscale reference PSNR:" + str(PSNRaccuracy(blurred, yb[0].cuda())))
        if show_images:
            show_tensor_as_image(blurred, "blurred")
        counter2 += 1


layer_output = {}
def get_activation(layer):
    def hook(model, input, output):
        layer_output[layer] = output.detach()
    return hook


def evaluateLayers(model, lr_dl, picture_numbers):
    model.eval()

    counter1 = 0
    for xb, yb in lr_dl:
        input = xb.cuda()[None, 0]
        result = model(input)
        clampedResult = result.clamp(0, 1)
        #print("input_shape:", str(input.size()))
        print("Model has run on picture ", picture_numbers[counter1])
        counter1 += 1
        return

def display_kernel_weights(kernel_weights):
    kernel_shape = kernel_weights.size()
    print("kernel_shape", str(kernel_shape))
    kernel_x_dim = math.floor(math.sqrt(kernel_shape[0]))
    kernel_y_dim = math.ceil(kernel_shape[0]/kernel_x_dim)

    kernel_array = np.zeros((kernel_shape[1], kernel_x_dim*kernel_shape[2] + kernel_x_dim, kernel_y_dim*kernel_shape[3] + kernel_y_dim))
    print("kernel_array shape:", str(kernel_array.shape))
    counter_kernel = 0
    for i in range(kernel_x_dim):
        for j in range(kernel_y_dim):
            add_val = kernel_weights[counter_kernel].detach().cpu().numpy()
            kernel_array[:,i*kernel_shape[2] + i:(i+1)*kernel_shape[2] + i, j*kernel_shape[3] + j:(j+1)*kernel_shape[3] + j] += add_val
            
            counter_kernel += 1

    #print("Kernel weights:", str(kernel_weights[2]))
    #show_tensor_as_image(kernel_weights[2].clamp(0,1))
    kernel_tensor = torch.from_numpy(kernel_array)
    show_tensor_as_image(kernel_tensor.clamp(0,1))



def display_feature_maps(layer_val):
    layer = layer_val.size()
    num_feature_maps = layer_val.size(dim=1)
    x_dim = math.floor(math.sqrt(num_feature_maps))
    y_dim = math.ceil(num_feature_maps/x_dim)

    #print("layer shape", str(layer), "x_dim", x_dim, "y_dim", y_dim)

    output_feature_array = np.zeros((x_dim*layer[2], y_dim*layer[3]))
    print("output_feature_tensor shape:", str(output_feature_array.shape))
    counter = 0
    for i in range(x_dim):
        for j in range(y_dim):
            if counter >= num_feature_maps:
                break
            
            add_val = layer_val[0][counter].cpu().numpy()
            output_feature_array[i*layer[2]:(i+1)*layer[2], j*layer[3]:(j+1)*layer[3]] += add_val
            
            counter += 1

    output_feature_tensor = torch.from_numpy(output_feature_array)
    show_tensor_as_image(output_feature_tensor)

# ------------------- CODE -------------------


def main():
    bs = 16 #16 for bignet, 128 for rest
    epochs = 1
    load_model = True

    data_aug = True
    


    train_dl, test_dl = loadFSRCNNdata(bs=bs)
    #test_dl = loadStanfordCars(upscaled=True)
    model, loss_func, optim, load_path = setupBigNet(load_model=load_model, lr=0.000005, load_path="BigNet16_Basic_AUG_Anneal")
    #testModel(model, loss_func, test_dl)
    #summary(model, input_size=(3,11,11))
    #showImage(model, "Stanford/Upscaled/sc8_Blurred", "Stanford/Cropped/sc8_crop", show_heatMap=True)
    #model = BicubicBaseline().cuda()
    fit(model, loss_func, opt=optim, train_dl=train_dl, valid_dl=test_dl, epochs=epochs, save_path=load_path, load_model=load_model, lr_scheduler = base_lr_schedule(), data_augmentation=data_aug)
    
    #fit2(model, loss_func=loss_func, opt=optim, trainset=train_dl, testset=test_dl, epochs=epochs, lr_scheduler=opt_lr_finding_schedule, data_aug=data_aug)


    lr_dl, upscaled_dl, picture_numbers = get_random_images_for_prediction(scale=3, listOfImages=["butterfly"])

    model.conv1.register_forward_hook(get_activation('conv1'))
    model.relu1.register_forward_hook(get_activation('relu1'))
    model.relu2.register_forward_hook(get_activation('relu2'))
    #model.relum4.register_forward_hook(get_activation('relum4'))
    #evaluateLayers(model, upscaled_dl, picture_numbers)
    

    #Choose layer to look at
    layer_val = layer_output['relu1']
    #display_feature_maps(layer_val)

    
    #Look at first layer of conv filters
    kernel_weights = model.conv1.weight
    #display_kernel_weights(kernel_weights)
    
    evaluateModel(model=model, lr_dl=lr_dl, upscaled_dl=upscaled_dl, picture_numbers=picture_numbers, show_images=True)

def loadDIV2K(upscaled=False):
    lr_path = "./Datasets/DIV2K_valid_HR/test_subset_Upscaled" if upscaled else "./Datasets/DIV2K_valid_HR/test_subset_LR"
    hr_path = "./Datasets/DIV2K_valid_HR/test_subset_cropped"
    return get_preprocessed_dataloader(lr_path, hr_path, bs=1)

def loadStanfordCars(upscaled=False):
    lr_path = "./Datasets/Stanford/Upscaled" if upscaled else "./Datasets/Stanford/LR"
    hr_path = "./Datasets/Stanford/Cropped"
    return get_preprocessed_dataloader(lr_path, hr_path, bs=1)

def showImage(model, lr_path, hr_path, show_heatMap=False):
    model.eval()
    toTensor = transforms.ToTensor()
    lr_file = Image.open("./Datasets/" + lr_path + ".png")
    hr_file = Image.open("./Datasets/" + hr_path + ".png")
    lr_image = toTensor(lr_file).cuda()
    hr_image = toTensor(hr_file).cuda()
    predicted = model(lr_image)
    show_tensor_as_image(lr_image)
    show_tensor_as_image(hr_image)
    show_tensor_as_image(predicted.clamp(0, 1))
    print("PSNR: ", PSNRaccuracy(predicted, hr_image))
    print("Baseline: ", PSNRaccuracy(lr_image, hr_image))

    if show_heatMap:
        heatMap = torch.abs(hr_image - predicted).detach().cpu().numpy()
        plt.imshow(heatMap.mean(axis=0), cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
        plt.colorbar()
        plt.show()
        
def test_lr():
    start_lr = 0.00008
    max_lr = 0.001
    lr = start_lr
    epochs = 100
    scheduler = cyclic_lr_schedule(max_lr, start_lr)
    lr_history = [] 
    epoch_hist = []
    for epoch in range(epochs):
        lr_history.append(lr)
        epoch_hist.append(epoch)
        lr = scheduler.update_learning_rate(epoch, 0, 0, epochs, lr)

    plt.figure()
    lines = []
    labels = []
    l, = plt.plot(epoch_hist, lr_history)
    lines.append(l)
    labels.append("Lr_hist")
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('Learning rate analysis')
    plt.legend(lines, labels, loc=(1, 0), prop=dict(size=14))
    plt.show()

if __name__ == "__main__":
    #test_lr()
    main()

