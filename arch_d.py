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

# create CNN model Arch-D according to the paper
class arch_d(torch.nn.Module):
    def __init__(self):
        super(arch_d, self).__init__()

        # use the convolutional layers from a pre-trained model
        self.conv = models.vgg16(True).features
        # define shared FC layer
        self.share = nn.Sequential()
        self.share.add_module("fc_share", nn.Linear(25088, 4096))     # 25088 -> 4096
        self.share.add_module("relu_share", nn.ReLU(True))
        self.share.add_module("dropout_share", nn.Dropout())
        # define output path for category prediction
        self.cate = nn.Sequential()
        self.cate.add_module("fc_cate_1", nn.Linear(4096, 1000))   # 4096 -> 1000
        self.cate.add_module("relu_cate_1", nn.ReLU(True))
        self.cate.add_module("dropout_cate_1", nn.Dropout())
        self.cate.add_module("fc_cate_out", nn.Linear(1000, 172))  # 1000 -> 172
        # define output path for ingredient prediction
        self.ingr = nn.Sequential()
        self.ingr.add_module("fc_ingr_1", nn.Linear(4096, 1000))   # 4096 -> 1000
        self.ingr.add_module("relu_ingr_1", nn.ReLU(True))
        self.ingr.add_module("dropout_ingr_1", nn.Dropout())
        self.ingr.add_module("fc_ingr_out", nn.Linear(1000, 353))  # 1000 -> 353

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(x.size(0), -1)
        x = self.share(x)
        cate_out = self.cate(x)
        ingr_out = self.ingr(x)
        return cate_out
        # return cate_out, ingr_out

def main():
    # initialize environment vars
    use_gpu = torch.cuda.is_available()


    # instantiate the modified CNN model
    model = arch_d()

    # freeze parameters in conv layers
    for param in model.conv.parameters():
        param.requires_grad = False 

    if use_gpu:
        model = model.cuda()

if __name__ == "__main__":
    main()