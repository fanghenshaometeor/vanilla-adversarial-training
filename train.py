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

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

import os
import ast
import copy
import time
import random
import argparse
import numpy as np

from utils import setup_seed, reduce_tensor
from attackers import pgd_attack

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== fix seed =============
setup_seed(666)

# ======== options ==============
parser = argparse.ArgumentParser(description='Vanilla & Adversarial Training Deep Neural Networks')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--model_dir',type=str,default='./save/',help='file path for saving model')
parser.add_argument('--logs_dir',type=str,default='./runs/',help='file path for logs')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--model',type=str,default='vgg16',help='model name')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=256,help='batch size for training (default: 256)')    
parser.add_argument('--epochs',type=int,default=200,help='number of epochs to train (default: 200)')
parser.add_argument('--local_rank',type=int,default=0,help='number of cpu threads')
# -------- enable adversarial training --------
parser.add_argument('--adv_train',type=ast.literal_eval,dest='adv_train',help='enable the adversarial training')
parser.add_argument('--adv_delay',type=int,default=10,help='epochs delay for adversarial training')
# -------- dataset params --------
parser.add_argument('--num_classes',type=int,default=10,help='number of classes')
args = parser.parse_args()

# ======== DDP init =============
torch.distributed.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)
num_gpus = torch.cuda.device_count()
print("-------- Let's use %d GPUs!"%num_gpus)

# ======== log writer init ======
if args.adv_train == True:
    writer = SummaryWriter(args.logs_dir+args.dataset+'-'+args.model+'-adv/')
else:
    writer = SummaryWriter(args.logs_dir+args.dataset+'-'+args.model+'/')

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
    elif args.dataset == 'CIFAR100':
        args.num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)
    
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, num_workers=8, pin_memory=True, sampler=DistributedSampler(trainset))
    testloader = data.DataLoader(testset, batch_size=args.batch_size, num_workers=8, pin_memory=True, sampler=DistributedSampler(testset))
    train_num, test_num = len(trainset), len(testset)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)
    print('---- #train : %d'%train_num)
    print('---- #test  : %d'%test_num)

    # ======== initialize net
    if args.model == 'vgg11':
        from model.vgg import vgg11_bn
        net = vgg11_bn(num_classes=args.num_classes).cuda()
    elif args.model == 'vgg13':
        from model.vgg import vgg13_bn
        net = vgg13_bn(num_classes=args.num_classes).cuda()
    elif args.model == 'vgg16':
        from model.vgg import vgg16_bn
        net = vgg16_bn(num_classes=args.num_classes).cuda()
    elif args.model == 'vgg19':
        from model.vgg import vgg19_bn
        net = vgg19_bn(num_classes=args.num_classes).cuda()
    elif args.model == 'wrn28x5':
        from model.wideresnet import wrn28
        net = wrn28(widen_factor=5, num_classes=args.num_classes).cuda()
    elif args.model == 'wrn28x10':
        from model.wideresnet import wrn28
        net = wrn28(widen_factor=10, num_classes=args.num_classes).cuda()
    else:
        assert False, "Unknow model : {}".format(args.model)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
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
    optimizer = optim.SGD(net.parameters(), lr=0.05*num_gpus, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60,120,160], gamma=0.1)
    
    print('-------- START TRAINING --------')

    for epoch in range(args.epochs):

        start = time.time()
        # -------- train
        train_epoch(net, trainloader, optimizer, criterion, epoch)

        # -------- validation
        if epoch % 20 == 0 or epoch == (args.epochs-1):
            corr_tr, corr_te = val(net, trainloader, testloader)
            acc_tr = corr_tr / train_num
            acc_te = corr_te / test_num
            acc_tr += reduce_tensor(torch.tensor(acc_tr).cuda(args.local_rank))
            acc_te += reduce_tensor(torch.tensor(acc_te).cuda(args.local_rank))

        scheduler.step()

        duration = time.time() - start

        # -------- save model & print info
        if args.local_rank == 0 and (epoch % 20 == 0 or epoch == (args.epochs-1)):
            checkpoint = {'state_dict': net.state_dict()}
            torch.save(checkpoint, args.model_path)
            print('Train/Test accuracy = %f/%f.'%(acc_tr, acc_te))

        # -------- print info.
        if args.local_rank == 0:
            print('Epoch %d/%d costs %fs :'%(epoch, args.epochs, duration))
            print('Current training model: ', args.model_path)
    

# ======== train  model ========
def train_epoch(net, trainloader, optim, criterion, epoch):
    
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
        running_loss_tr = running_loss_tr + reduce_tensor(loss.data).item()

        # -------- print info. at the last epoch
        if args.local_rank == 0 and batch_idx == (len(trainloader)-1):
            avg_loss_tr = running_loss_tr / len(trainloader)

            # -------- record
            writer.add_scalar('loss-train', avg_loss_tr)

            # -------- print in terminal
            print('Epoch %d/%d CLEAN samples:'%(epoch, args.epochs))
            print('     CROSS ENTROPY loss = %f.'%avg_loss_tr)

        
        # -------- backprop. & update
        loss.backward()
        optim.step()

        if args.adv_train and epoch >= args.adv_delay:
            # -------- training with adversarial examples
            net.eval()
            perturbed_data = pgd_attack(net, b_data, b_label, eps=0.031, alpha=0.01, iters=7)
            net.train()

            logits_adv = net(perturbed_data)
            loss_adv = criterion(logits_adv, b_label)
            running_loss_tr_adv = running_loss_tr_adv + reduce_tensor(loss_adv.data).item()

            # -------- print info. at the last epoch
            if args.local_rank == 0 and batch_idx == (len(trainloader)-1):
                avg_loss_tr_adv = running_loss_tr_adv / len(trainloader)

                # -------- record
                writer.add_scalar('loss-train-adv', avg_loss_tr_adv)

                # -------- print in terminal
                print('Epoch %d/%d ADVERSARIAL samples:'%(epoch, args.epochs))
                print('     CROSS ENTROPY loss = %f.'%avg_loss_tr_adv)

            # -------- backprop. & update again
            optim.zero_grad()
            loss_adv.backward()
            optim.step()

    return

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