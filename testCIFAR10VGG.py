"""
Created on Wed Feb 26 2020

@author: fanghenshao
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torchvision import datasets, transforms

import random
import argparse
import numpy as np

from vgg import *
from utils import *

# -------- fix data type ----------------
torch.set_default_tensor_type(torch.FloatTensor)

# -------- test model ---------------
def test(args, net, trainloader, testloader, device):
    
    net.eval()
    
    correct_train = 0
    correct_test = 0
    
    with torch.no_grad():
        
        # -------- compute the accs. of train, test set
        for test in testloader:
            images, labels = test
            images = images.to(device)
            labels = labels.to(device)
            
            logits = net(images)
            logits = logits.detach()
            _, predicted = torch.max(logits.data, 1)
            
            correct_test += (predicted == labels).sum().item()
        
        for train in trainloader:
            images, labels = train
            images = images.to(device)
            labels = labels.to(device)
            
            logits = net(images)
            logits = logits.detach()
            _, predicted = torch.max(logits.data, 1)
            
            correct_train += (predicted == labels).sum().item()  
    
    return correct_train, correct_test

# -------- main function
def main():

    # ======== parameter settings =======
    parser = argparse.ArgumentParser(description='Test VGG on CIFAR10')
    # -------- file param. --------------
    parser.add_argument('--data-folder', type=str, default='/media/Disk1/KunFang/data/CIFAR10/', metavar='DATAFOLDER',
                        help='file path for data')
    parser.add_argument('--model-folder', type=str, default='model/', metavar='MODELFOLDER',
                        help='file path for model')
    # -------- training param. ----------
    parser.add_argument('--batch-size', type=int, default=256, metavar='BATCHSIZE',
                        help='input batch size for training (default: 512)')
    args = parser.parse_args()


    # -------- fix seed -----------------
    setup_seed(666)
    
    # -------- device -------------------
    device = torch.device('cuda:3')

    # ======== load MNIST data set 28 x 28  =============
    args.dataset = 'CIFAR10'
    args.model_name = args.dataset + '-VGG.pth'
    # args.model_name = args.dataset + '-VGG-adv.pth'
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(root=args.data_folder, train=True, download=True, transform=transform_test)
    testset = datasets.CIFAR10(root=args.data_folder, train=False, download=True, transform=transform_test)

    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    train_num, test_num = len(trainset), len(testset)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)
    print('---- #train : %d' % train_num)
    print('---- #test  : %d' % test_num)

    # ======== initialize net and load parameters
    checkpoint = torch.load(args.model_folder+args.model_name, map_location=torch.device("cpu"))
    net = vgg16_bn()
    net.load_state_dict(checkpoint['state_dict'])
    net = net.to(device)
    net.eval()

    print('-------- START TESTING --------')
    corr_tr, corr_te = test(args, net, trainloader, testloader, device)
    acc_tr = corr_tr / train_num
    acc_te = corr_te / test_num
    print('Train acc. = %f; Test acc. = %f.' % (acc_tr, acc_te))

   
# -------- start point
if __name__ == '__main__':
    main()
    
