"""
Created on Mon Feb 24 2020

@author: fanghenshao

"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torchvision import datasets, transforms

import os
import ast
import copy
import random
import argparse
import numpy as np

from utils import setup_seed
from attackers import pgd_attack

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== fix seed =============
setup_seed(666)

# ======== options ==============
parser = argparse.ArgumentParser(description='Training Deep Neural Networks')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--model_dir',type=str,default='save/',help='file path for saving model')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--model',type=str,default='vgg16',help='model name')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=256,help='batch size for training (default: 256)')    
parser.add_argument('--epochs',type=int,default=200,help='number of epochs to train (default: 200)')
parser.add_argument('--gpu_id',type=str,default='0',help='gpu device index')
# -------- enable adversarial training --------
parser.add_argument('--adv_train',type=ast.literal_eval,dest='adv_train',help='enable the adversarial training')
args = parser.parse_args()

# ======== GPU device ========
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# -------- main function
def main():
    
    # ======== data set preprocess =============
    # ======== mean-variance normalization is removed
    if args.dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    elif args.dataset == 'STL10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])
        trainset = datasets.STL10(root=args.data_dir, split='train', transform=transform_train, download=True)
        testset = datasets.STL10(root=args.data_dir, split='test', transform=transform_test, download=True)
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)
    
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    train_num, test_num = len(trainset), len(testset)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)
    print('---- #train : %d'%train_num)
    print('---- #test  : %d'%test_num)

    # ======== initialize net
    if args.model == 'vgg11':
        from model.vgg import vgg11_bn
        net = vgg11_bn().cuda()
    elif args.model == 'vgg13':
        from model.vgg import vgg13_bn
        net = vgg13_bn().cuda()
    elif args.model == 'vgg16':
        from model.vgg import vgg16_bn
        net = vgg16_bn().cuda()
    elif args.model == 'vgg19':
        from model.vgg import vgg19_bn
        net = vgg19_bn().cuda()
    elif args.model == 'resnet18':
        from model.resnet import ResNet18
        net = ResNet18().cuda()
    elif args.model == 'resnet20':
        from model.resnet_v1 import resnet20
        net = resnet20().cuda()
    elif args.model == 'modela':
        from model.modela import ModelA
        net = ModelA().cuda()
    else:
        assert False, "Unknow model : {}".format(args.model)
    if args.adv_train:
        args.model_path = args.model_dir+args.dataset+'-'+args.model+'-adv.pth'
    else:
        args.model_path = args.model_dir+args.dataset+'-'+args.model+'.pth'
    print('-------- MODEL INFORMATION --------')
    print('---- model:      '+args.model)
    print('---- adv. train: '+str(args.adv_train))
    print('---- saved path: '+args.model_path)

    # ======== set criterions & optimizers
    criterion = nn.CrossEntropyLoss()
    if args.model == 'vgg11' or args.model == 'vgg13' or args.model == 'vgg16' or args.model == 'vgg19':
        args.epochs = 200
        optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60,120,160], gamma=0.1)
    elif args.model == 'resnet18' or args.model == 'resnet20':
        args.epochs = 350
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[150,250],gamma=0.1)
    elif args.model == 'modela':
        args.epochs = 200 
        optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60,120,160], gamma=0.1)
    
    print('-------- START TRAINING --------')

    for epoch in range(args.epochs):

        # -------- train
        loss_tr, loss_tr_adv = train(net, trainloader, testloader, optimizer, criterion)

        # -------- validation
        corr_tr, corr_te = val(net, trainloader, testloader)
        acc_tr = corr_tr / train_num
        acc_te = corr_te / test_num

        scheduler.step()

        # -------- save model
        checkpoint = {'state_dict': net.state_dict()}
        torch.save(checkpoint, args.model_path)

        if args.adv_train:
            print('Epoch %d: train loss = %f; adv. train loss = %f; train acc. = %f; test acc. = %f.' % (epoch, loss_tr, loss_tr_adv, acc_tr, acc_te))
        else:
            print('Epoch %d: train loss = %f; train acc. = %f; test acc. = %f.' % (epoch, loss_tr, acc_tr, acc_te))
    

# ======== train  model ========
def train(net, trainloader, testloader, optim, criterion):
    
    net.train()
        
    running_loss_tr = 0.0
    avg_loss_tr = 0.0
    running_loss_tr_adv = 0.0
    avg_loss_tr_adv = 0.0

    
    for batch_idx, (b_data, b_label) in enumerate(trainloader):
        
        # -------- move to gpu
        b_data, b_label = b_data.cuda(), b_label.cuda()
        
        # -------- set zero param. gradients
        optim.zero_grad()
              
        # -------- feed noise to the network
        logits = net(b_data)
        
        # -------- compute loss
        loss = criterion(logits, b_label)
        
        running_loss_tr = running_loss_tr + loss.item()
        if batch_idx == (len(trainloader)-1):
            avg_loss_tr = running_loss_tr / len(trainloader)
        
        # -------- backprop. & update
        loss.backward()
        optim.step()

        if args.adv_train:
            # -------- training with adversarial examples
            net_copy = copy.deepcopy(net)
            net_copy.eval()
            perturbed_data = pgd_attack(net_copy, b_data, b_label, eps=0.013, alpha=0.01, iters=7)
            logits_adv = net(perturbed_data)
            loss_adv = criterion(logits_adv, b_label)

            running_loss_tr_adv = running_loss_tr_adv + loss_adv.item()
            if batch_idx == (len(trainloader)-1):
                avg_loss_tr_adv = running_loss_tr_adv / len(trainloader)

            # -------- backprop. & update again
            optim.zero_grad()
            loss_adv.backward()
            optim.step()

    return avg_loss_tr, avg_loss_tr_adv

# ======== evaluate model ========
def val(net, trainloader, testloader):
    
    net.eval()
    
    correct_train = 0
    correct_test = 0
    
    with torch.no_grad():
        
        # -------- compute the accs. of train, test set
        for test in testloader:
            images, labels = test
            images, labels = images.cuda(), labels.cuda()
            
            logits = net(images)
            logits = logits.detach()
            _, predicted = torch.max(logits.data, 1)
            
            correct_test += (predicted == labels).sum().item()
        
        for train in trainloader:
            images, labels = train
            images, labels = images.cuda(), labels.cuda()
            
            logits = net(images)
            logits = logits.detach()
            _, predicted = torch.max(logits.data, 1)
            
            correct_train += (predicted == labels).sum().item()  
    
    return correct_train, correct_test


# ======== startpoint
if __name__ == '__main__':
    main()