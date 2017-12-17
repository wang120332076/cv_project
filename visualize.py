from my_util import visualize_model
import numpy as np
import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms
from my_set import my_set
import matplotlib.pyplot as plt

def main(model_file, test_all=False):
    # ----- Dataset related ----- #
    test_all = False
    # initialize environment vars
    if test_all:
        data_dirname = '../vireo172/vireo172_sets'
    else:
        data_dirname = '../vireo172/vireo172_lite'
    sets_name = ['train', 'val']
    label_filename = '../vireo172/SplitAndIngreLabel/FoodList.txt'
    ingr_filename = '../vireo172/SplitAndIngreLabel/IngreLabel.txt'
    use_gpu = torch.cuda.is_available()

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
    image_datasets = {x: my_set(os.path.join(data_dirname, x), ingr_label,
                                             data_transforms[x])
                    for x in sets_name}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in sets_name}
    dataset_sizes = {x: len(image_datasets[x]) for x in sets_name}
    class_names = image_datasets['train'].classes
    # create label list
    with open(label_filename, 'r', encoding="utf-8") as f:
        cate_labels = f.read().splitlines()
    class_names = [cate_labels[(int(x)-1)] for x in class_names]

    model = torch.load(model_file)

    visualize_model(model, dataloaders, class_names, use_gpu, 6)

if __name__ == '__main__':
    plt.ion()

    args = sys.argv
    
    if len(args) != 2:
        print('Error: wrong input parameters')
        quit(1)

    main(args[1])
    plt.ioff()

    # quit(0)