"""
Created on Mon Mar 09 2020

@author: fanghenshao
"""

from __future__ import print_function
from tabnanny import check


import torch
import torch.nn as nn

import os
import sys
import ast
import time
import argparse

import numpy as np

from utils import setup_seed, get_parameter_number
from utils import get_datasets, get_model
from utils import AverageMeter, accuracy
from utils import Logger

from advertorch.attacks import GradientSignAttack
from advertorch.attacks import LinfPGDAttack
from advertorch.attacks import CarliniWagnerL2Attack

from autoattack import AutoAttack

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== options ==============
parser = argparse.ArgumentParser(description='Attack Deep Neural Networks')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--output_dir',type=str,default='./output/',help='folder to store output')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
parser.add_argument('--model_path',type=str,default='./save/CIFAR10-VGG.pth',help='saved model path')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=256,help='batch size for training (default: 256)')
# -------- attack param. ----------
parser.add_argument('--test_eps', default=8., type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=20, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=2., type=float, help='step size of attack during testing')
parser.add_argument('--attack_type',type=str,default='fgsm',help='attack method')
args = parser.parse_args()

# -------- initialize output store dir.
if 'adv' in args.model_path:
    if not os.path.exists(os.path.join(args.output_dir,args.dataset,args.arch+'-adv')):
        os.makedirs(os.path.join(args.output_dir,args.dataset,args.arch+'-adv'))
    args.output_path = os.path.split(args.model_path)[-1].replace(".pth", "-"+args.attack_type.upper()+".log")
    args.output_path = os.path.join(args.output_dir,args.dataset,args.arch+'-adv',args.output_path)
else:
    if not os.path.exists(os.path.join(args.output_dir,args.dataset,args.arch)):
        os.makedirs(os.path.join(args.output_dir,args.dataset,args.arch))
    args.output_path = os.path.split(args.model_path)[-1].replace(".pth", "-"+args.attack_type.upper()+".log")
    args.output_path = os.path.join(args.output_dir,args.dataset,args.arch,args.output_path)
sys.stdout = Logger(filename=args.output_path,stream=sys.stdout)


# -------- main function
def main():

    # ======== fix seed =============
    setup_seed(666)
    
    # ======== data set preprocess =============
    trainloader, testloader = get_datasets(args)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)

    # ======== initialize net
    net = get_model(args)
    net = net.cuda()
    checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    print('-------- MODEL INFORMATION --------')
    print('---- architecture: '+args.arch)
    print('---- saved path  : '+args.model_path)
    print('---- # of param  : ')
    print('---- ', get_parameter_number(net))
    if 'best' in args.model_path:
        print('---- best robust acc. achieved at epoch-%d.'%checkpoint['best-epoch'])
    
    args.test_eps /= 255.
    args.test_gamma /= 255.
 
    if args.attack_type == 'None':
        print('-------- START TESTING --------')
        acc_tr = val(net, trainloader)
        acc_te = val(net, testloader)
        print('---- Train/Test acc. = %.3f/%.2f.' % (acc_tr, acc_te))

    elif args.attack_type == 'pgd':
        print('-------- START ATTACKING --------')
        print('-------- ADVERSARY INFORMATION --------')
        print('---- PGD attack with %d/255 step size, %d iterations and bound %d/255.'%(args.test_gamma*255, args.test_step, args.test_eps*255))
        # --------
        acc_attack = attack(net, testloader)
        print('Attacked PGD acc. = %.2f'%acc_attack)
    
    elif args.attack_type == 'fgsm':
        print('-------- START ATTACKING --------')
        print('-------- ADVERSARY INFORMATION --------')
        print('---- FGSM attack with bound %d/255.'%(args.test_eps*255))
        # --------
        acc_attack = attack(net, testloader)
        print('Attacked FGSM acc. = %.2f'%acc_attack)
    
    elif args.attack_type == 'cw':
        print('-------- START ATTACKING --------')
        print('-------- ADVERSARY INFORMATION --------')
        print('---- C&W attack with default settings in AdverTorch.')
        # --------
        acc_attack = attack(net, testloader)
        print('Attacked C&W acc. = %.2f'%acc_attack)
    
    elif args.attack_type == 'square':
        print('-------- START ATTACKING --------')
        print('-------- ADVERSARY INFORMATION --------')
        print('---- SQUARE attack with default settings in AutoAttack.')
        # --------
        acc_attack = attack(net, testloader)
        print('Attacked SQUARE acc. = %.2f'%acc_attack)

    print('-------- FINISHED.')
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


# -------- attack model --------
def attack(net, testloader):

    net.eval()
    top1 = AverageMeter()

    if args.attack_type == 'pgd':
        adversary = LinfPGDAttack(net, loss_fn=nn.CrossEntropyLoss(), eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    elif args.attack_type == 'fgsm':
        adversary = GradientSignAttack(net, loss_fn=nn.CrossEntropyLoss(), eps=args.test_eps, clip_min=0.0, clip_max=1.0, targeted=False)
    elif args.attack_type == 'cw':
        adversary = CarliniWagnerL2Attack(net, num_classes=args.num_classes)
    elif args.attack_type == 'square':
        adversary = AutoAttack(net, norm='Linf', eps=args.test_eps, version='standard')
        adversary.attacks_to_run = ['square']

    for test in testloader:
        image, label = test
        image, label = image.cuda(), label.cuda()

        # generate adversarial examples
        if args.attack_type == 'square':
            perturbed_image = adversary.run_standard_evaluation(image, label, bs=image.size(0))
        else:
            perturbed_image = adversary.perturb(image, label)

        # re-classify
        logits = net(perturbed_image).detach().float()
        prec1 = accuracy(logits.data, label)[0]
        top1.update(prec1.item(), image.size(0))

    return top1.avg


# -------- start point
if __name__ == '__main__':
    main()