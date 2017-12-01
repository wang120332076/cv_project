# pylint: disable=C0103,C0111

from __future__ import print_function, division
import time
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

# Function for calculate micro f1 and macro f1
def ingredient_accuracy(predict, truth):
    # input should be N*353 tensor.
    predict = predict > 0.5
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

# define training process
def train_model(model, dataloaders, dataset_sizes, use_gpu, stat_filename, \
                criterion1, criterion2, optimizer, scheduler, num_epochs=25):
    LAMBDA = 0.2
    COMPENSATE = 353

    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # write out status file
        with open(stat_filename, 'a') as f:
            f.write('\nCurrent epoch {}/{}'.format(epoch, num_epochs - 1) + '\n')
            f.write('-' * 10 + '\n')
            te = time.time() - since
            f.write('Time elapsed: {:.0f}m {:.0f}s'.format( \
                    te // 60, te % 60) + '\n')

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
                    sum_R += TP / (TP + FN)
                if (TP+FP != 0):
                    sum_P += TP / (TP + FP)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            out_stat1 = '{} Loss: {:.4f} Acc (category): {:.4f}'.format( \
                        phase, epoch_loss, epoch_acc)
            print(out_stat1)

            # calculate the accuracy for ingredient prediction result
            if (sum_TP+sum_FP != 0):
                epoch_P1 = sum_TP/(sum_TP+sum_FP)
            if (sum_TP+sum_FN != 0):
                epoch_R1 = sum_TP/(sum_TP+sum_FN)
            if (epoch_P1 + epoch_R1 != 0):
                epoch_micro = 2 * epoch_P1 * epoch_R1 / (epoch_P1 + epoch_R1)

            epoch_R2 = sum_R / dataset_sizes[phase] * dataloaders[phase].batch_size
            epoch_P2 = sum_P / dataset_sizes[phase] * dataloaders[phase].batch_size
            if (epoch_P2 + epoch_R2 != 0):
                epoch_macro = 2 * epoch_P2 * epoch_R2/ (epoch_P2 + epoch_R2)

            out_stat2 = '{} Acc (ingredients): Micro-F1: {:.4f} Macro-F1: {:.4f}'.format( \
                        phase, epoch_micro, epoch_macro)
            print(out_stat2)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

            # write out status file
            with open(stat_filename, 'a') as f:
                f.write(out_stat1 + '\n')
                f.write(out_stat2 + '\n')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, dataloaders, class_names, use_gpu, num_images=6):
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