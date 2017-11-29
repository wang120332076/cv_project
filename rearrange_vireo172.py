import os
from shutil import copy2

src_PATH = '../vireo172/ready_chinese_food'
dst_PATH = '../vireo172/vireo172_sets'

train_NAME = '../vireo172/SplitAndIngreLabel/TR.txt'
val_NAME = '../vireo172/SplitAndIngreLabel/VAL.txt'
test_NAME = '../vireo172/SplitAndIngreLabel/TE.txt'

with open(train_NAME, 'r') as f:
    lines = f.read().splitlines()
for xx in lines:
    copy2(src_PATH + xx, dst_PATH + '/train' + xx)

with open(val_NAME, 'r') as f:
    lines = f.read().splitlines()
for xx in lines:
    copy2(src_PATH + xx, dst_PATH + '/val' + xx)

with open(test_NAME, 'r') as f:
    lines = f.read().splitlines()
for xx in lines:
    copy2(src_PATH + xx, dst_PATH + '/test' + xx)
