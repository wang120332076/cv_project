# pylint: disable=C0103

from __future__ import print_function, division
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import my_model
import my_util as util
from my_set import my_set

def main(model_name, train_all, epoch_num=20, save_name=None):
    print('Using model - \t\t\'%s\'' % model_name)
    print('Train all layers - \t\'%s\'' % train_all)
    print('Epoch number - \t\t\'%d\'' % epoch_num)

    # ----- Dataset related ----- #
    # initialize environment vars
    if train_all:
        data_dirname = '../vireo172/vireo172_sets'
    else:
        data_dirname = '../vireo172/vireo172_lite'
    sets_name = ['train', 'val']
    label_filename = '../vireo172/SplitAndIngreLabel/FoodList.txt'
    ingr_filename = '../vireo172/SplitAndIngreLabel/IngreLabel.txt'
    use_gpu = torch.cuda.is_available()

    # initialize runtime status file
    stat_filename = './run_stat'
    if os.path.exists(stat_filename):
        print('Backing up existing status log')
        os.rename(stat_filename, stat_filename + '.old')
    with open(stat_filename, 'w') as f:
        f.write('Using model - \t\t\'%s\'' % model_name)
        f.write('\n')
        f.write('Train all layers - \t\'%s\'' % train_all)
        f.write('\n')
        f.write('Epoch number - \t\t\'%d\'' % epoch_num)
        f.write('\n')

    # read in ingredient label list
    with open(ingr_filename, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    ingr_label = {}
    for ll in lines:
        str_l = ll.strip().split()
        name = str_l[0]
        str_l = str_l[1:]
        int_l = [0 if int(x) == (-1) else 1 for x in str_l]
        out = torch.FloatTensor(int_l)
        ingr_label[name] = out

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

    # create dataset objects using pyTorch provided methods
    if train_all:
        batch_s = 50
    else:
        batch_s = 4
    image_datasets = {x: my_set(os.path.join(data_dirname, x), ingr_label,
                                             data_transforms[x])
                    for x in sets_name}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_s,
                                                shuffle=True, num_workers=4)
                for x in sets_name}
    dataset_sizes = {x: len(image_datasets[x]) for x in sets_name}
    class_names = image_datasets['train'].classes
    # create label list
    with open(label_filename, 'r', encoding="utf-8") as f:
        cate_labels = f.read().splitlines()
    # labels = [labels[int(x)-1] for x in class_names]

    # ----- CNN related ----- #
    # instantiate the modified CNN model
    if model_name == 'vgg16':
        model = my_model.arch_d_vgg16(True)
        optim_params = list(model.share.parameters()) + list(model.cate.parameters()) + \
                       list(model.ingr.parameters())
    elif model_name == 'plain_vgg16':
        model = my_model.plain_vgg16(True)
        optim_params = list(model.cate.parameters())
    elif model_name == 'plain_resnet152':
        model = my_model.plain_resnet152(True)
        optim_params = list(model.share.parameters()) + list(model.cate.parameters()) + \
                       list(model.conv.fc.parameters())
    elif model_name == 'plain_densenet161':
        model = my_model.plain_densenet161(True)
        optim_params = list(model.share.parameters()) + list(model.cate.parameters()) + \
                       list(model.conv.classifier.parameters())
    elif model_name == 'plain_inception':
        model = my_model.plain_inception(True)
        optim_params = list(model.share.parameters()) + list(model.cate.parameters()) + \
                       list(model.conv.fc.parameters())
                       
    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # then unfreeze the part which needs training
    for param in optim_params:
        param.requires_grad = True

    # check if we can use GPU
    if use_gpu:
        model = model.cuda()

    # prepare for training
    L1 = nn.NLLLoss()
    L2 = nn.BCELoss()

    # Observe that only parameters of layers that are being optimized
    optimizer_conv = optim.SGD(optim_params, lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every step_s epochs
    if train_all:
        step_s = 4
    else:
        step_s = 7
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_s, gamma=0.1)

    # start training
    is_plain = (model_name.split('_')[0] == 'plain')
    if is_plain:
        model = util.train_model_cate(model, dataloaders, dataset_sizes, use_gpu, stat_filename, \
                                      L1, optimizer_conv, exp_lr_scheduler, epoch_num)
    else:
        model = util.train_model(model, dataloaders, dataset_sizes, use_gpu, stat_filename, \
                                 L1, L2, optimizer_conv, exp_lr_scheduler, epoch_num)

    # show results
    # visualize_model(model, dataloaders, use_gpu)

    # save trained model
    if save_name is None:
        str_a = model_name + '_'
        str_b = ('1' if train_all else '0') + '_'
        str_c = str(epoch_num)
        save_name = str_a + str_b + str_c
    save_name = './' + save_name
    if os.path.exists(save_name):
        print('Backing up existing saved model')
        os.rename(save_name, save_name + '.old')
    print('Saving trained model to: \'%s\'' % save_name)
    torch.save(model.cpu(), save_name)

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    args = sys.argv
    # print(args)

    # check input param#
    if(len(args) not in [4, 5]):
        print('Error: wrong input parameters')
        quit(1)
    # check model name
    models_list = ['vgg16', 'plain_vgg16', 'plain_resnet152', \
                   'plain_inception', 'plain_densenet161']
    if(args[1] not in models_list):
        print('Error: wrong model name')
        quit(2)
    # call main function
    if (len(args) == 4):
        main(args[1], bool(int(args[2])), int(args[3]))
    else:
        main(args[1], bool(int(args[2])), int(args[3]), args[4])

    quit(0)
