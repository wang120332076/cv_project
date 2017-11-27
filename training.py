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
from arch_d import arch_d
from my_set import my_set

plt.ion()

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
with open(ingr_filename, 'r') as f:
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
with open(label_filename, 'r') as f:
    cate_labels = f.read().splitlines()
# labels = [labels[int(x)-1] for x in class_names]

# define training process
def train_model(model, criterion1, criterion2, optimizer, scheduler, num_epochs=25):
    LAMBDA = 0.02
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
                loss = loss1 + LAMBDA * loss2

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds.data == cate_l.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc (category): {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

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
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            print(class_names[preds[j]])
            print(labels[preds[j]])
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

################### Arch-D based on VGG16
if os.path.exists(save_name):
    model = torch.load(save_name)
    os.rename(save_name, save_name+'.old')
else:
    # instantiate the modified CNN model
    model = arch_d()
    # freeze parameters in conv layers
    for param in model.conv.parameters():
        param.requires_grad = False

# check if we can use GPU
if use_gpu:
    model = model.cuda()

# prepare for training
L1 = nn.NLLLoss()
L2 = nn.BCELoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optim_params = list(model.share.parameters()) + list(model.cate.parameters()) + list(model.ingr.parameters())
optimizer_conv = optim.SGD(optim_params, lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# start training
model = train_model(model, L1, L2, optimizer_conv, exp_lr_scheduler, 15)

# show results
# visualize_model(model)

# save trained model
torch.save(model, save_name)


plt.ioff()
plt.show()
