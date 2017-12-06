# pylint: disable=C0103

from __future__ import print_function, division
import torch
import torch.nn as nn
from torchvision import models

# use plain VGG16 on categorization for performance comparison
class plain_vgg16(torch.nn.Module):
    def __init__(self, preTrained=False):
        super(plain_vgg16, self).__init__()

        self.conv = models.vgg16(preTrained).features
        self.cate = nn.Sequential()
        self.cate.add_module("fc_1", nn.Linear(25088, 4096)) # 25088 -> 4096
        self.cate.add_module("relu_1", nn.ReLU(True))
        self.cate.add_module("dropout_1", nn.Dropout())
        self.cate.add_module("fc_2", nn.Linear(4096, 4096))  # 4096 -> 4096
        self.cate.add_module("relu_2", nn.ReLU(True))
        self.cate.add_module("dropout_2", nn.Dropout())
        self.cate.add_module("fc_3", nn.Linear(4096, 172))   # 4096 -> 172
        self.cate.add_module("softmax_cate", nn.LogSoftmax())

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(x.size(0), -1)
        cate_out = self.cate(x)
        return cate_out

class plain_resnet152(torch.nn.Module):
    def __init__(self, preTrained=False):
        super(plain_resnet152, self).__init__()

        # instantiate resnet-152
        self.conv = models.resnet152(preTrained)
        # define shared FC layer
        self.conv.fc = nn.Linear(2048, 2048)
        self.share = nn.Sequential()
        self.share.add_module("relu_share", nn.ReLU(True))
        self.share.add_module("dropout_share", nn.Dropout())
        # define output path for category prediction
        self.cate = nn.Sequential()
        self.cate.add_module("cate_fc_1", nn.Linear(2048, 1024))  # 2048 -> 1024
        self.cate.add_module("cate_relu_1", nn.ReLU(True))
        self.cate.add_module("cate_dropout_1", nn.Dropout())
        self.cate.add_module("cate_fc_2", nn.Linear(1024, 172))   # 1024 -> 172
        self.cate.add_module("softmax_cate", nn.LogSoftmax())

    def forward(self, x):
        x = self.conv.forward(x)
        x = self.share(x)
        x = x.view(x.size(0), -1)
        cate_out = self.cate(x)
        return cate_out

class plain_densenet161(torch.nn.Module):
    def __init__(self, preTrained=False):
        super(plain_densenet161, self).__init__()

        # instantiate resnet-152
        self.conv = models.densenet161(preTrained)
        # define shared FC layer
        self.conv.classifier = nn.Linear(2208, 2048)
        self.share = nn.Sequential()
        self.share.add_module("relu_share", nn.ReLU(True))
        self.share.add_module("dropout_share", nn.Dropout())
        # define output path for category prediction
        self.cate = nn.Sequential()
        self.cate.add_module("cate_fc_1", nn.Linear(2048, 1024))  # 2048 -> 1024
        self.cate.add_module("cate_relu_1", nn.ReLU(True))
        self.cate.add_module("cate_dropout_1", nn.Dropout())
        self.cate.add_module("cate_fc_2", nn.Linear(1024, 172))   # 1024 -> 172
        self.cate.add_module("softmax_cate", nn.LogSoftmax())

    def forward(self, x):
        x = self.conv.forward(x)
        x = self.share(x)
        x = x.view(x.size(0), -1)
        cate_out = self.cate(x)
        return cate_out

class plain_inception(torch.nn.Module):
    def __init__(self, preTrained=False):
        super(plain_inception, self).__init__()

        # instantiate inception_v3
        self.conv = models.inception_v3(preTrained)
        self.conv.aux_logits = False
        # define shared FC layer
        self.conv.fc = nn.Linear(2048, 2048)
        self.share = nn.Sequential()
        self.share.add_module("relu_share", nn.ReLU(True))
        self.share.add_module("dropout_share", nn.Dropout())
        # define output path for category prediction
        self.cate = nn.Sequential()
        self.cate.add_module("cate_fc_1", nn.Linear(2048, 1024))  # 2048 -> 1024
        self.cate.add_module("cate_relu_1", nn.ReLU(True))
        self.cate.add_module("cate_dropout_1", nn.Dropout())
        self.cate.add_module("cate_fc_2", nn.Linear(1024, 172))   # 1024 -> 172
        self.cate.add_module("softmax_cate", nn.LogSoftmax())

    def forward(self, x):
        x = self.conv.forward(x)
        x = self.share(x)
        x = x.view(x.size(0), -1)
        cate_out = self.cate(x)
        return cate_out

# create CNN model Arch-D according to the paper
class arch_d_vgg16(torch.nn.Module):
    def __init__(self, preTrained=False):
        super(arch_d_vgg16, self).__init__()

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
    def __init__(self, preTrained=False):
        super(arch_d_res50, self).__init__()

        # use the convolutional layers from a pre-trained model
        self.conv = models.resnet50(preTrained)
        # define shared FC layer
        self.conv.fc = nn.Linear(2048, 2048)
        self.share = nn.Sequential()
        self.share.add_module("relu_share", nn.ReLU(True))
        self.share.add_module("dropout_share", nn.Dropout())
        # define output path for category prediction
        self.cate = nn.Sequential()
        self.cate.add_module("cate_fc_1", nn.Linear(2048, 1024))  # 2048 -> 1024
        self.cate.add_module("cate_relu_1", nn.ReLU(True))
        self.cate.add_module("cate_dropout_1", nn.Dropout())
        self.cate.add_module("cate_fc_2", nn.Linear(1024, 172))   # 1024 -> 172
        self.cate.add_module("softmax_cate", nn.LogSoftmax())
        # define output path for ingredient prediction
        self.ingr = nn.Sequential()
        self.ingr.add_module("ingr_fc_1", nn.Linear(2048, 1024))  # 2048 -> 1024
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