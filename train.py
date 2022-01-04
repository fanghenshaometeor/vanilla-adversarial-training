"""
Created on Mon Feb 24 2020

@author: fanghenshao

"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import os
import ast
import copy
import time
import random
import argparse
import numpy as np

from utils import setup_seed
from utils import get_datasets, get_model
from utils import AverageMeter, accuracy

from advertorch.attacks import LinfPGDAttack
from advertorch.context import ctx_noparamgrad_and_eval

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== options ==============
parser = argparse.ArgumentParser(description='Vanilla & Adversarial Training Deep Neural Networks')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--model_dir',type=str,default='./save/',help='file path for saving model')
parser.add_argument('--logs_dir',type=str,default='./runs/',help='file path for logs')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=256,help='batch size for training (default: 256)')    
parser.add_argument('--lr_base',type=float,default=0.05,help='learning rate (default: 0.05)')
parser.add_argument('--epochs',type=int,default=200,help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq',type=int,default=20,help='model save frequency (default: 20 epoch)')
# -------- enable adversarial training --------
parser.add_argument('--adv_train',type=ast.literal_eval,dest='adv_train',help='enable the adversarial training')
parser.add_argument('--train_eps', default=8., type=float, help='epsilon of attack during training')
parser.add_argument('--train_step', default=10, type=int, help='itertion number of attack during training')
parser.add_argument('--train_gamma', default=2., type=float, help='step size of attack during training')
parser.add_argument('--test_eps', default=8., type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=20, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=2., type=float, help='step size of attack during testing')
args = parser.parse_args()

# ======== log writer init ======
if args.adv_train == True:
    writer = SummaryWriter(os.path.join(args.logs_dir,args.dataset,args.arch+'-adv/'))
else:
    writer = SummaryWriter(os.path.join(args.logs_dir,args.dataset,args.arch+'/'))

# -------- main function
def main():

    # ======== fix random seed ========
    setup_seed(666)
    
    # ======== get data set =============
    trainloader, testloader = get_datasets(args)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)

    # ======== initialize net
    net = get_model(args)
    net = net.cuda()
    print('-------- MODEL INFORMATION --------')
    print('---- architecture: '+args.arch)
    print('---- adv.   train: '+str(args.adv_train))

    # ======== set criterions & optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60,120,160], gamma=0.1)

    # ======== 
    args.train_eps /= 255.
    args.train_gamma /= 255.

    print('-------- START TRAINING --------')
    for epoch in range(1, args.epochs+1):

        # -------- train
        print('Training(%d/%d)...'%(epoch, args.epochs))
        train_epoch(net, trainloader, optimizer, criterion, epoch)

        # -------- validation
        print('Validating...')
        acc_te = val(net, testloader)
        acc_tr = val(net, trainloader)

        # ---- record
        valacc = {}
        valacc['train'], valacc['test'] = acc_tr, acc_te
        writer.add_scalars('valacc', valacc, epoch)
        print('     Train/Test accuracy = %.2f/%.2f.'%(acc_tr, acc_te))
        
        scheduler.step()

        # -------- save model & print info
        if not os.path.exists(os.path.join(args.model_dir,args.dataset,args.arch)):
            os.makedirs(os.path.join(args.model_dir,args.dataset,args.arch))

        if (epoch == 1 or epoch % args.save_freq == 0 or epoch == args.epochs):
            checkpoint = {'state_dict': net.state_dict()}
            if args.adv_train:
                args.model_path = 'epoch%d'%epoch+'-adv.pth'
            else:
                args.model_path = 'epoch%d'%epoch+'.pth'
            torch.save(checkpoint, os.path.join(args.model_dir,args.dataset,args.arch,args.model_path))

        # -------- print info.
        print('Current training %s on data set %s.'%(args.arch, args.dataset))
        print('===========================================')
    

# ======== train  model ========
def train_epoch(net, trainloader, optim, criterion, epoch):
    
    net.train()
        
    batch_time = AverageMeter()
    losses = AverageMeter()

    if args.adv_train:
        adversary = LinfPGDAttack(net, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)


    end = time.time()
    for _, (b_data, b_label) in enumerate(trainloader):
        
        # -------- move to gpu
        b_data, b_label = b_data.cuda(), b_label.cuda()

        
        if args.adv_train:
            # -------- training with adversarial examples
            with ctx_noparamgrad_and_eval(net):
                perturbed_data = adversary.perturb(b_data, b_label)
            logits = net(perturbed_data)
            loss = criterion(logits, b_label)
        else:
            # -------- feed clean data to the network
            logits = net(b_data)
            loss = criterion(logits, b_label)

        # -------- backprop. & update
        optim.zero_grad()
        loss.backward()
        optim.step()

        # -------- update info
        loss = loss.float()
        losses.update(loss.item(), b_data.size(0))
        # ----
        batch_time.update(time.time()-end)
        end = time.time()

    print('Epoch %d costs %fs :'%(epoch, batch_time.sum))

    # -------- record & print in terminal
    trainstats = {}
    trainstats['loss'] = losses.avg
    if args.adv_train:
        print('     CROSS ENTROPY loss on ADV.  TRAIN = %f.'%(losses.avg))
    else:
        print('     CROSS ENTROPY loss on CLEAN TRAIN = %f.'%(losses.avg))
    writer.add_scalars('trainstats', trainstats, epoch)
        
    return

# ======== evaluate model ========
def val(net, dataloader):
    
    net.eval()
    top1 = AverageMeter()
        
    # clean
    with torch.no_grad():
        
        # -------- compute the accs. of train, test set
        for _, test in enumerate(dataloader):
            images, labels = test
            images, labels = images.cuda(), labels.cuda()
            
            logits = net(images).detach().float()

            prec1 = accuracy(logits.data, labels)[0]
            top1.update(prec1.item(), images.size(0))
            
    return top1.avg



# ======== startpoint
if __name__ == '__main__':
    main()