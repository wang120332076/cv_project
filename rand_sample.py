import os
from shutil import copy2
from numpy import random

src_PATH = '../vireo172/ready_chinese_food'
dst_PATH = '../vireo172/vireo172_lite'

train_NAME = '../vireo172/SplitAndIngreLabel/TR.txt'
val_NAME = '../vireo172/SplitAndIngreLabel/VAL.txt'
test_NAME = '../vireo172/SplitAndIngreLabel/TE.txt'

train_N = 64
val_N = 16
total_N = train_N + val_N

total_list = [[] for x in range(172)]
# read train list
with open(train_NAME, 'r') as f:
    lines = f.read().splitlines()
for xx in lines:
    temp = xx.strip('/').split('/')
    ind = int(temp[0]) - 1
    total_list[ind].append(xx) 
# read val list
with open(val_NAME, 'r') as f:
    lines = f.read().splitlines()
for xx in lines:
    temp = xx.strip('/').split('/')
    ind = int(temp[0]) - 1
    total_list[ind].append(xx) 
#read test list
with open(test_NAME, 'r') as f:
    lines = f.read().splitlines()
for xx in lines:
    temp = xx.strip('/').split('/')
    ind = int(temp[0]) - 1
    total_list[ind].append(xx)
len_list = [len(x) for x in total_list]

#judge the path exist or not
for ii in range(173):
    if ii and not os.path.exists(dst_PATH+'/train/' + str(ii)) :
        os.mkdir(dst_PATH+'/train/' + str(ii))
    if ii and not os.path.exists(dst_PATH+'/val/' + str(ii)) :
        os.mkdir(dst_PATH+'/val/' + str(ii))

# generate smaller training and validation sets
for f_list in total_list:
    N = len(f_list)
    # get a sample of images with number of N
    samp = random.permutation(N)
    samp_train = samp[0:train_N]
    samp_val = samp[train_N:total_N]
    for ii in samp_train:
        fn = f_list[ii]
        copy2(src_PATH + fn, dst_PATH + '/train' + fn)
    for ii in samp_val:
        fn = f_list[ii]
        copy2(src_PATH + fn, dst_PATH + '/val' + fn)


