# ************************************************************
# Author : Dang-Khoa
# Source : https://github.com/meliketoy/fine-tuning.pytorch
#
# Description : main-avito.py
# The main code for training regression network using Avito Data
# ***********************************************************

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import config as cf
import torchvision
import time
import copy
import os
import sys
import argparse
import math

from datasets import AvitoImage, preprocess_avito
from torchvision import datasets, models, transforms
from networks import *
from torch.autograd import Variable
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--finetune', '-f', action='store_true', help='Fine tune pretrained model')
parser.add_argument('--addlayer','-a',action='store_true', help='Add additional layer in fine-tuning')
parser.add_argument('--resetClassifier', '-r', action='store_true', help='Reset classifier')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--fold', default=5, type=float, help='number of folds')

args = parser.parse_args()

# Phase 1 : Data Upload
print('\n[Phase 1] : Data Preperation')

fold = preprocess_avito(cf.train_csv_file, cf.n_folds, cf.fold_idx, cf.is_train)
images = fold['images']
probs = fold['probs']
print("| Data Summary")
print("| #samples = {}".format(len(images)))
# filter out items which don't have image files
# data = data[data.image.notnull()]
# data = shuffle(data)
# nsamples = len(data)
# nsamples_fold = nsamples//args.fold
# images = {
#     'train': data[0:nsamples_fold*(args.fold-1)].image.values,
#     'val': data[nsamples_fold*(args.fold-1):].image.values
# }

# probs = {
#     'train': data[0:nsamples_fold*(args.fold-1)].deal_probability.values,
#     'val': data[nsamples_fold*(args.fold-1):].deal_probability.values
# }
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean, cf.std)
    ]),
    'val': transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean, cf.std)
    ]),
}

data_dir = cf.aug_base
dataset_dir = cf.data_base.split("/")[-1] + os.sep
print("| Preparing model trained on %s dataset..." %(cf.data_base.split("/")[-1]))

dsets = {
    x : AvitoImage(images[x], probs[x], data_dir, data_transforms[x])
    for x in ['train', 'val']
}

dset_loaders = {
    x : DataLoader(dsets[x], batch_size=cf.batch_size,
                   shuffle=(x=='train'), num_workers=20)
    for x in ['train', 'val']
}

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()
if (use_gpu):
    print('| Use GPU')

# Phase 2 : Model setup
print('\n[Phase 2] : Model setup')

def getNetwork(args):
    if (args.net_type == 'alexnet'):
        net = models.alexnet(pretrained=args.finetune)
        file_name = 'alexnet'
    elif (args.net_type == 'vggnet'):
        if(args.depth == 11):
            net = models.vgg11(pretrained=args.finetune)
        elif(args.depth == 13):
            net = models.vgg13(pretrained=args.finetune)
        elif(args.depth == 16):
            net = models.vgg16(pretrained=args.finetune)
        elif(args.depth == 19):
            net = models.vgg19(pretrained=args.finetune)
        else:
            print('Error : VGGnet should have depth of either [11, 13, 16, 19]')
            sys.exit(1)
        file_name = 'vgg-%s' %(args.depth)
    elif (args.net_type == 'squeezenet'):
        net = models.squeezenet1_0(pretrained=args.finetune)
        file_name = 'squeeze'
    elif (args.net_type == 'resnet'):
        net = resnet(args.finetune, args.depth)
        file_name = 'resnet-%s' %(args.depth)
    else:
        print('Error : Network should be either [alexnet / squeezenet / vggnet / resnet]')
        sys.exit(1)

    return net, file_name

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Test only option
if (args.testOnly):
    print("| Loading checkpoint model for test phase...")
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    print('| Loading '+file_name+".t7...")
    checkpoint = torch.load('./checkpoint/'+dataset_dir+'/'+file_name+'.t7')
    model = checkpoint['model']

    if use_gpu:
        print('| Use GPU')
        model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        # cudnn.benchmark = True

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    testsets = datasets.ImageFolder(cf.test_dir, data_transforms['val'])

    testloader = torch.utils.data.DataLoader(
        testsets,
        batch_size = 1,
        shuffle = False,
        num_workers=1
    )

    print("\n[Phase 3 : Inference on %s]" %cf.test_dir)
    for batch_idx, (inputs, targets) in enumerate(testloader):#dset_loaders['val']):
        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)

        # print(outputs.data.cpu().numpy()[0])
        softmax_res = softmax(outputs.data.cpu().numpy()[0])

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total
    print("| Test Result\tAcc@1 %.2f%%" %(acc))

    sys.exit(0)

# Training model
def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=cf.num_epochs):
    global dataset_dir
    since = time.time()
    float_info = np.finfo(float)
    best_model, best_acc = model, float_info.max

    print('\n[Phase 3] : Training Model')
    print('| Training Epochs = %d' %num_epochs)
    print('| Initial Learning Rate = %f' %args.lr)
    print('| Optimizer = SGD')
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer, lr = lr_scheduler(optimizer, epoch)
                print('\n=> Training Epoch #%d, LR=%f' %(epoch+1, lr))
                model.train(True)
            else:
                model.train(False)
                model.eval()

            running_loss, running_corrects, tot = 0.0, 0, 0

            for batch_idx, batch in enumerate(dset_loaders[phase]):
                if use_gpu:
                    inputs = Variable(batch['image'].cuda())
                    labels = Variable(batch['prob'].cuda())
                else:
                    inputs = Variable(batch['image'])
                    labels = Variable(batch['prob'])
                labels = labels.unsqueeze(1)
                optimizer.zero_grad()

                # Forward Propagation
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                loss = torch.sqrt(loss)

                # Backward Propagation
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Statistics
                running_loss += loss.item()
                tot += labels.size(0)

                if (phase == 'train'):
                    sys.stdout.write('\r')
                    sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d] Loss %.4f Running Loss %.4f\t'
                            %(epoch+1, num_epochs, batch_idx+1,
                                math.ceil(len(dsets[phase])//cf.batch_size),
                              loss.item(), running_loss))
                    sys.stdout.flush()
                    sys.stdout.write('\r')

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc  = epoch_loss

            if (phase == 'val'):
                print('\n| Validation Epoch #%d\t\t\tLoss %.4f\tMSE %.2f%%'
                    %(epoch+1, loss.item(), epoch_acc))

                if epoch_acc < best_acc :#and epoch > 80:
                    print('| Saving Best model...\t\t\tMSE %.2f%%' %(100.*epoch_acc))
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)
                    state = {
                        'model': best_model,
                        'acc':   epoch_acc,
                        'epoch':epoch,
                    }
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    save_point = './checkpoint/'+dataset_dir
                    if not os.path.isdir(save_point):
                        os.mkdir(save_point)
                    print("Save path = {}".format(save_point))
                    torch.save(state, save_point+file_name+cf.aux_name+'.t7')

    time_elapsed = time.time() - since
    print('\nTraining completed in\t{:.0f} min {:.0f} sec'. format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc\t{:.2f}%'.format(best_acc*100))

    return best_model

def exp_lr_scheduler(optimizer, epoch, init_lr=args.lr, weight_decay=args.weight_decay, lr_decay_epoch=cf.lr_decay_epoch):
    lr = init_lr * (0.5**(epoch // lr_decay_epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = weight_decay

    return optimizer, lr

model_ft, file_name = getNetwork(args)

if(args.resetClassifier):
    print('| Reset final classifier...')
    if(args.addlayer):
        sys.exit("Do not support this feature now")
        print('| Add features of size %d' %cf.feature_size)
        num_ftrs = model_ft.fc.in_features
        feature_model = list(model_ft.fc.children())
        feature_model.append(nn.Linear(num_ftrs, cf.feature_size))
        feature_model.append(nn.BatchNorm1d(cf.feature_size))
        feature_model.append(nn.ReLU(inplace=True))
        feature_model.append(nn.Linear(cf.feature_size, len(dset_classes)))
        model_ft.fc = nn.Sequential(*feature_model)
    else:
        if(args.net_type == 'alexnet' or args.net_type == 'vggnet'):
            num_ftrs = model_ft.classifier[6].in_features
            feature_model = list(model_ft.classifier.children())
            feature_model.pop()
            feature_model.append(nn.Linear(num_ftrs, 1))
            model_ft.classifier = nn.Sequential(*feature_model)
            # sys.exit('Only support resnet for now')
        elif(args.net_type == 'resnet'):
            print('| Add regression layer')
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 1)

if use_gpu:
    model_ft = model_ft.cuda()
    model_ft = torch.nn.DataParallel(model_ft, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if __name__ == "__main__":
    criterion = nn.MSELoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=cf.num_epochs)
