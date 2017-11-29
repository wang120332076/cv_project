# pylint: disable=C0103

from __future__ import print_function, division
import io
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import models, transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from arch_d import arch_d, arch_d_res50
from my_set import my_set

plt.ion()

# Function for calculate micro f1 and macro f1
def ingredient_accuracy(predict, truth):
    # input should be 4*353 tensors.
    #predict = predict>0.5
    _, ind = torch.topk(predict, 3)        # grab 3 ingr with max score as prediction
    predict = predict * 0
    for x in range(predict.size()[0]):
        for y in ind[x,:]:
            predict[x, y] = 1
    predict = predict.float()
    TP = torch.sum(predict * truth)
    TN = torch.sum((1 - predict) * (1 - truth))
    FP = torch.sum(predict * (1 - truth))
    FN = torch.sum((1 - predict) * truth)
    return TP, TN, FP, FN

# Function for visualizing images
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# define transformation for each set of images
norm_mean = np.array([0.485, 0.456, 0.406])
norm_std = np.array([0.229, 0.224, 0.225])
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
    'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
}

# define dataset(vireo-172) path
data_dirname = '../vireo172/vireo172_lite'
sets_name = ['train', 'val']
label_filename = '../vireo172/SplitAndIngreLabel/FoodList.txt'
ingr_filename = '../vireo172/SplitAndIngreLabel/IngreLabel.txt'
save_name = './arch_d.pth'
use_gpu = torch.cuda.is_available()

# read in ingredient label list
with open(ingr_filename, 'r', encoding="utf-8") as f:
    lines = f.readlines()

ingr_label = {}
for ll in lines:
    str_l = ll.strip().split()
    name = str_l[0]
    str_l = str_l[1:]
    int_l = [0 if int(x)==(-1) else 1 for x in str_l]
    out = torch.FloatTensor(int_l)
    ingr_label[name] = out

# create dataset objects using pyTorch provided methods
image_datasets = {x: my_set(os.path.join(data_dirname, x), ingr_label,
                                         data_transforms[x])
                  for x in sets_name}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=2)
               for x in sets_name}
dataset_sizes = {x: len(image_datasets[x]) for x in sets_name}
class_names = image_datasets['train'].classes
# create label list
with open(label_filename, 'r', encoding="utf-8") as f:
    cate_labels = f.read().splitlines()
# labels = [labels[int(x)-1] for x in class_names]

# define training process
def train_model(model, criterion1, criterion2, optimizer, scheduler, num_epochs=25):
    LAMBDA = 0.2
    COMPENSATE = 353
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            #intialize parameters for calculating ingredients' accuracy
            sum_TP = 0; sum_FP = 0; sum_FN = 0; sum_R = 0; sum_P = 0; 

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, cate_l, ingr_l = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    cate_l = Variable(cate_l.cuda())
                    ingr_l = Variable(ingr_l.cuda())
                else:
                    inputs = Variable(inputs)
                    cate_l = Variable(cate_l)
                    ingr_l = Variable(ingr_l)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                cate_pred, ingr_pred = model(inputs)
                _, preds = torch.max(cate_pred, 1)
                loss1 = criterion1(cate_pred, cate_l)
                loss2 = criterion2(ingr_pred, ingr_l)
                loss = loss1 + LAMBDA * COMPENSATE * loss2

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics for loss and accuracy
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds.data == cate_l.data)

                TP, TN, FP, FN = ingredient_accuracy(ingr_pred.data, ingr_l.data)
                sum_TP += TP
                sum_FP += FP
                sum_FN += FN
                if (TP+FN != 0):
                    sum_R += TP/(TP+FN)
                if (TP+FP != 0):
                    sum_P += TP/(TP+FP)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc (category): {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # calculate the accuracy for ingredient prediction result
            if (sum_TP+sum_FP != 0):
                epoch_P1 = sum_TP/(sum_TP+sum_FP)
            if (sum_TP+sum_FN != 0):
                epoch_R1 = sum_TP/(sum_TP+sum_FN)
            epoch_micro = 2*epoch_P1*epoch_R1/(epoch_P1+epoch_R1)

            epoch_R2 = sum_R/dataset_sizes[phase]
            epoch_P2 = sum_P/dataset_sizes[phase]
            epoch_macro = 2*epoch_P2*epoch_R2/(epoch_P2+epoch_R2)

            print('{} Ingr acc: micro f1: {:.4f} macro f1: {:.4f}'.format(
                phase, epoch_micro, epoch_macro))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, cate_l, ingr_l = data
        if use_gpu:
            inputs, cate_l, ingr_l = Variable(inputs.cuda()), Variable(cate_l.cuda()), Variable(ingr_l.cuda())
        else:
            inputs, cate_l, ingr_l = Variable(inputs), Variable(cate_l), Variable(ingr_l)

        cate_pred, ingr_pred = model(inputs)
        _, preds = torch.max(cate_pred.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            print(class_names[preds[j]])
            print(cate_l[preds[j]])
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

################### Arch-D based on VGG16
if os.path.exists(save_name):
    model = torch.load(save_name)
    os.rename(save_name, save_name+'.old')
else:
    # instantiate the modified CNN model
    model = arch_d(True)
    # freeze parameters in conv layers
    for param in model.conv.parameters():
        param.requires_grad = False
    #for param in model.conv.fc.parameters():
    #    param.requires_grad = True

# check if we can use GPU
if use_gpu:
    model = model.cuda()

# prepare for training
L1 = nn.NLLLoss()
L2 = nn.BCELoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
# optim_params = model.parameters()
optim_params = list(model.share.parameters()) + list(model.cate.parameters()) + list(model.ingr.parameters())
# optim_params = list(model.share.parameters()) + list(model.cate.parameters()) + \
               # list(model.ingr.parameters()) + list(model.conv.fc.parameters())
optimizer_conv = optim.SGD(optim_params, lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# start training
model = train_model(model, L1, L2, optimizer_conv, exp_lr_scheduler, 20)

# show results
# visualize_model(model)

# save trained model
torch.save(model.cpu(), save_name)


plt.ioff()
plt.show()
