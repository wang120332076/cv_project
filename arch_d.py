# pylint: disable=C0103

from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import models, transforms, datasets
import numpy as np

# create CNN model Arch-D according to the paper
class arch_d(torch.nn.Module):
    def __init__(self, preTrained=False):
        super(arch_d, self).__init__()

        # use the convolutional layers from a pre-trained model
        self.conv = models.vgg16(preTrained).features
        # define shared FC layer
        self.share = nn.Sequential()
        self.share.add_module("fc_share", nn.Linear(25088, 4096)) # 25088 -> 4096
        self.share.add_module("relu_share", nn.ReLU(True))
        self.share.add_module("dropout_share", nn.Dropout())
        # define output path for category prediction
        self.cate = nn.Sequential()
        self.cate.add_module("cate_fc_1", nn.Linear(4096, 4096))  # 4096 -> 4096
        self.cate.add_module("cate_1relu_", nn.ReLU(True))
        self.cate.add_module("cate_dropout_1", nn.Dropout())
        self.cate.add_module("cate_fc_2", nn.Linear(4096, 172))   # 4096 -> 172
        self.cate.add_module("softmax_cate", nn.LogSoftmax())
        # define output path for ingredient prediction
        self.ingr = nn.Sequential()
        self.ingr.add_module("ingr_fc_1", nn.Linear(4096, 1024))  # 4096 -> 1024
        self.ingr.add_module("ingr_relu_1", nn.ReLU(True))
        self.ingr.add_module("ingr_dropout_1", nn.Dropout())
        self.ingr.add_module("ingr_fc_2", nn.Linear(1024, 353))   # 1024 -> 353
        self.ingr.add_module("sigmoid_ingr", nn.Sigmoid())

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(x.size(0), -1)
        x = self.share(x)
        cate_out = self.cate(x)
        ingr_out = self.ingr(x)
        return cate_out, ingr_out


# recreate Arch-D based on ResNet-50
class arch_d_res50(torch.nn.Module):
    def __init__(self):
        super(arch_d_res50, self).__init__()

        # use the convolutional layers from a pre-trained model
        self.conv = models.resnet50(True)
        # define shared FC layer
        self.conv.fc = nn.Linear(2048,2048)
        self.share = nn.Sequential()
        self.share.add_module("relu_share", nn.ReLU(True))
        self.share.add_module("dropout_share", nn.Dropout())
        # define output path for category prediction
        self.cate = nn.Sequential()
        self.cate.add_module("cate_fc_1", nn.Linear(2048, 1024))  # 4096 -> 4096
        self.cate.add_module("cate_1relu_", nn.ReLU(True))
        self.cate.add_module("cate_dropout_1", nn.Dropout())
        self.cate.add_module("cate_fc_2", nn.Linear(1024, 172))   # 4096 -> 172
        self.cate.add_module("softmax_cate", nn.LogSoftmax())
        # define output path for ingredient prediction
        self.ingr = nn.Sequential()
        self.ingr.add_module("ingr_fc_1", nn.Linear(2048, 1024))  # 4096 -> 1024
        self.ingr.add_module("ingr_relu_1", nn.ReLU(True))
        self.ingr.add_module("ingr_dropout_1", nn.Dropout())
        self.ingr.add_module("ingr_fc_2", nn.Linear(1024, 353))   # 1024 -> 353
        self.ingr.add_module("sigmoid_ingr", nn.Sigmoid())

    def forward(self, x):
        x = self.conv.forward(x)
        x = self.share(x)
        cate_out = self.cate(x)
        ingr_out = self.ingr(x)
        return cate_out, ingr_out
