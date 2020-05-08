"""
Created on Mon Feb 24 2020

@author: fanghenshao

"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torchvision import datasets, transforms

import copy
import random
import argparse
import numpy as np

from utils import *
from vgg import *

# -------- fix data type ----------------
torch.set_default_tensor_type(torch.FloatTensor)

# -------- PGD attack --------
def pgd_attack(args, net, image, label, eps, alpha=0.01, iters=7, random_start=True, d_min=0, d_max=1):

    perturbed_image = image.clone()
    perturbed_image.requires_grad = True

    image_max = image + eps
    image_min = image - eps
    image_max.clamp_(d_min, d_max)
    image_min.clamp_(d_min, d_max)

    if random_start:
        with torch.no_grad():
            perturbed_image.data = image + perturbed_image.uniform_(-1*eps, eps)
            perturbed_image.data.clamp_(d_min, d_max)
    
    for _ in range(iters):
        logits = net(perturbed_image)
        loss = F.cross_entropy(logits, label)
        if perturbed_image.grad is not None:
            perturbed_image.grad.data.zero_()
        
        loss.backward()
        data_grad = perturbed_image.grad.data

        with torch.no_grad():
            perturbed_image.data += alpha * torch.sign(data_grad)
            perturbed_image.data = torch.max(torch.min(perturbed_image, image_max), image_min)
    perturbed_image.requires_grad = False
    
    return perturbed_image

# -------- train  model 
def train(args, net, trainloader, testloader, optim, criterion, device):
    
    net.train()
        
    running_loss_tr = 0.0
    avg_loss_tr = 0.0
    running_loss_tr_adv = 0.0
    avg_loss_tr_adv = 0.0
    running_loss_te = 0.0
    avg_loss_te = 0.0

    for batch_idx, (b_data, b_label) in enumerate(testloader):
        # -------- move to gpu
        b_data, b_label = b_data.to(device), b_label.to(device)
                      
        # -------- feed noise to the network
        logits = net(b_data)
        
        # -------- compute loss
        loss = criterion(logits, b_label)
        
        running_loss_te = running_loss_te + loss.item()
        if batch_idx == (len(testloader)-1):
            avg_loss_te = running_loss_te / len(testloader)
    
    for batch_idx, (b_data, b_label) in enumerate(trainloader):
        
        # -------- move to gpu
        b_data, b_label = b_data.to(device), b_label.to(device)
        
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

        # -------- training with adversarial examples
        net_copy = copy.deepcopy(net)
        net_copy.eval()
        perturbed_data = pgd_attack(args, net_copy, b_data, b_label, eps=0.013, alpha=0.01, iters=7)
        logits_adv = net(perturbed_data)
        loss_adv = criterion(logits_adv, b_label)

        running_loss_tr_adv = running_loss_tr_adv + loss_adv.item()
        if batch_idx == (len(trainloader)-1):
            avg_loss_tr_adv = running_loss_tr_adv / len(trainloader)

        optim.zero_grad()
        loss_adv.backward()
        optim.step()

    return avg_loss_tr, avg_loss_te, avg_loss_tr_adv

# -------- evaluate model ---------------
def val(args, net, trainloader, testloader, device):
    
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
    parser = argparse.ArgumentParser(description='Adversarial Training VGG on CIFAR10')
    # -------- file param. --------------
    parser.add_argument('--data-folder', type=str, default='/media/Disk1/KunFang/data/CIFAR10/', metavar='DATAFOLDER',
                        help='file path for data')
    parser.add_argument('--model-folder', type=str, default='model/', metavar='MODELFOLDER',
                        help='file path for model')
    parser.add_argument('--log-folder', type=str, default='log/', metavar='LOGFOLDER',
                        help='file path for log')
    # -------- training param. ----------
    parser.add_argument('--batch-size', type=int, default=256, metavar='BATCHSIZE',
                        help='input batch size for training (default: 512)')    
    parser.add_argument('--epochs', type=int, default=200, metavar='EPOCH',
                        help='number of epochs to train (default: 100000)')
    args = parser.parse_args()
    
    # -------- fix seed -----------------
    setup_seed(666)
    
    # -------- device -------------------
    device = torch.device('cuda:1')

    # ======== load CIFAR10 data set 32 x 32  =============
    args.dataset = 'CIFAR10'
    args.model_name = args.dataset + '-VGG-adv.pth'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    trainset = datasets.CIFAR10(root=args.data_folder, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=args.data_folder, train=False, download=True, transform=transform_test)
    
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    train_num, test_num = len(trainset), len(testset)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)
    print('---- #train : %d'%train_num)
    print('---- #test  : %d'%test_num)

    # ======== initialize net
    net = vgg16_bn()
    net = net.to(device)

    # ======== set criterions & optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60,120,160], gamma=0.1)

    # ======== initialize variables
    losses_train = np.array([])
    losses_test = np.array([])
    losses_adv = np.array([])
    accs_train = np.array([])
    accs_test = np.array([])

    print('-------- START TRAINING --------')

    for epoch in range(args.epochs):

        # -------- train
        loss_tr, loss_te, loss_tr_adv = train(args, net, trainloader, testloader, optimizer, criterion, device)

        # -------- validation
        corr_tr, corr_te = val(args, net, trainloader, testloader, device)
        acc_tr = corr_tr / train_num
        acc_te = corr_te / test_num

        scheduler.step()

        # -------- save info
        losses_train = np.append(losses_train, loss_tr)
        losses_test = np.append(losses_test, loss_te)
        losses_adv = np.append(losses_adv, loss_tr_adv)
        accs_train = np.append(accs_train, acc_tr)
        accs_test = np.append(accs_test, acc_te)

        # -------- save model
        checkpoint = {'state_dict': net.state_dict()}
        torch.save(checkpoint, args.model_folder+args.model_name)

        print('Epoch %d: train loss = %f; test loss = %f; adv. train loss = %f; train acc. = %f; test acc. = %f.' % (epoch, loss_tr, loss_te, loss_tr_adv, acc_tr, acc_te))
    
    np.save(args.log_folder+'losses-train',losses_train)
    np.save(args.log_folder+'losses-test',losses_test)
    np.save(args.log_folder+'losses-adv',losses_adv)
    np.save(args.log_folder+'acc-train',accs_train)
    np.save(args.log_folder+'acc-test',accs_test)

# -------- startpoint
if __name__ == '__main__':
    main()