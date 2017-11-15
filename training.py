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
use_gpu = torch.cuda.is_available()

# create dataset objects using pyTorch provided methods
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dirname, x),
                                          data_transforms[x])
                  for x in sets_name}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=2)
               for x in sets_name}
dataset_sizes = {x: len(image_datasets[x]) for x in sets_name}
class_names = image_datasets['train'].classes
# create label list
with open(label_filename, 'r') as f:
    labels = f.read().splitlines()
# labels = [labels[int(x)-1] for x in class_names]

# define training process
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
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
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
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
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

################### VGG-16
# # create pre-trained VGG16
# vgg16 = models.vgg16(pretrained = True)
# for param in vgg16.parameters():
#     param.requires_grad = False     # freeze parameters

# # modify FC layers
# vgg16.classifier._modules['6'] = nn.Linear(4096, 172)
# if use_gpu:
#     vgg16 = vgg16.cuda()

# # prepare for training
# criterion = nn.CrossEntropyLoss()

# # Observe that only parameters of final layer are being optimized as
# # opoosed to before.
# optimizer_conv = optim.SGD(vgg16.classifier._modules['6'].parameters(),
#                            lr=0.001, momentum=0.9)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# # start training
# vgg16 = train_model(vgg16, criterion, optimizer_conv, exp_lr_scheduler, 25)

# # show results
# visualize_model(vgg16)

################### ResNet-18
resnet = models.resnet18(True)
for param in resnet.parameters():
    param.requires_grad = False     # freeze parameters

resnet.fc = nn.Linear(512, 172)
if use_gpu:
    resnet = resnet.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(resnet.fc.parameters(),
                           lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
resnet = train_model(resnet, criterion, optimizer_conv, exp_lr_scheduler, 25)

visualize_model(resnet)

# example code for visualizing images in dataset
# im, lab_n = next(iter(dataloaders['train']))
# out = torchvision.utils.make_grid(im)
# imshow(out, [labels[x] for x in lab_n])

plt.ioff()
plt.show()