"""
Created on Mon Mar 09 2020

@author: fanghenshao
"""

from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torchvision import datasets, transforms

import argparse

from utils import *
from vgg import *

# -------- args initialization --------
def args_init():

    parser = argparse.ArgumentParser(description='Attack VGG on CIFAR10')
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
    

    args = parser.parse_args()
    return args

# -------- FGSM attack --------
def fgsm_attack(args, net, image, label, epsilon):
    image.requires_grad = True

    logits = net(image)
    loss = F.cross_entropy(logits, label)
    net.zero_grad()
    loss.backward()

    # collect data grad    
    perturbed_image = image + epsilon*image.grad.data.sign()
    # here we cannot clip the perturbed image into [0,1] 
    # since the original image has been preprocessed by mean-var normalization,
    # which results in that the pixel range is no longer [0,1].
    # Therefore we clip the perturbed image into the new pixel range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # print("original  image max value = %f; min value = %f."%(image.max(), image.min()))
    # print("perturbed image max value = %f; min value = %f."%(perturbed_image.max(), perturbed_image.min()))
    perturbed_image = torch.clamp(perturbed_image, image.min().data, image.max().data)
    return perturbed_image

# -------- PGD attack --------
def pgd_attack(args, net, image, label, eps, alpha=0.01, iters=7):
    ori_image = image.data
    min_val = image.min().data
    max_val = image.max().data
    for _ in range(iters):
        image.requires_grad = True

        logits = net(image)
        loss = F.cross_entropy(logits, label)
        net.zero_grad()
        loss.backward()

        adv_image = image + alpha * image.grad.data.sign()
        eta = torch.clamp(adv_image - ori_image, min=-eps, max=eps)

        image = torch.clamp(ori_image + eta, min=min_val, max=max_val).detach_()
    
    return image


# -------- attack model --------
def attack(args, net, testloader, epsilon, attackType):

    correct = 0

    net.eval()

    # loop all examples in test set
    for test in testloader:
        image, label = test
        image, label = image.to(args.device), label.to(args.device) 

        # generate adversarial examples
        if attackType == "FGSM":
            perturbed_image = fgsm_attack(args, net, image, label, epsilon)
        elif attackType == "PGD":
            perturbed_image = pgd_attack(args, net, image, label, epsilon)

        # re-classify
        logits = net(perturbed_image)

        _, final_pred = torch.max(logits.data, 1)
        correct = correct + (final_pred == label).sum().item() 
    return correct

# -------- test model ---------------
def test(args, net, trainloader, testloader):
    
    net.eval()
    
    correct_train = 0
    correct_test = 0
    
    with torch.no_grad():
        
        # -------- compute the accs. of train, test set
        for test in testloader:
            images, labels = test
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            logits = net(images)
            logits = logits.detach()
            _, predicted = torch.max(logits.data, 1)
            
            correct_test += (predicted == labels).sum().item()
        
        for train in trainloader:
            images, labels = train
            images = images.to(args.device)
            labels = labels.to(args.device)
            
            logits = net(images)
            logits = logits.detach()
            _, predicted = torch.max(logits.data, 1)
            
            correct_train += (predicted == labels).sum().item()  
    
    return correct_train, correct_test

# -------- main function --------
def main():

    # ======== parametes initialization =======
    args = args_init()
    args.device = 'cuda:3'

    # ======== fix random seed ========
    setup_seed(666)

    # ======== load CIFAR10 data set 32 x 32  =============
    args.dataset = 'CIFAR10'
    args.model_name = args.dataset + '-VGG.pth'
    # args.model_name = args.dataset + '-VGG-adv.pth'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(root=args.data_folder, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=args.data_folder, train=False, download=True, transform=transform)
    
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    train_num, test_num = len(trainset), len(testset)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+ args.dataset)
    print('---- #train : %d'%train_num)
    print('---- #test  : %d'%test_num)
    args.epochs = 200

    # ======== VGG16_bn initialization ========
    checkpoint = torch.load(args.model_folder+args.model_name, map_location=torch.device("cpu"))
    net = vgg16_bn()
    net.load_state_dict(checkpoint['state_dict'])
    net = net.to(args.device)
    net.eval()
    print('-------- MODEL INFORMATION --------')
    print('---- model: VGG16 with batch normalization')

    print('-------- START TESTING --------')
    corr_tr, corr_te = test(args, net, trainloader, testloader)
    acc_tr, acc_te = corr_tr / train_num, corr_te / test_num
    print('Train acc. = %f; Test acc. = %f.' % (acc_tr, acc_te))

    print('-------- START FGSM ATTACK --------')
    fgsm_epsilons = [.05, .1, .15, .2, .25, .3]
    print('---- EPSILONS: ', fgsm_epsilons)
    for eps in fgsm_epsilons:
        print('---- current eps = %.2f...'%eps)
        correct_te_fgsm = attack(args, net, testloader, eps, "FGSM")
        acc_te_fgsm = correct_te_fgsm / float(test_num)
        print('Attacked test acc. = %f.'%acc_te_fgsm)

    print('-------- START PGD ATTACK -------')
    pgd_epsilon = 0.031         # 8/255=0.031
    print('---- EPSILON = %f'%pgd_epsilon)
    corr_te_pgd = attack(args, net, testloader, pgd_epsilon, "PGD")
    acc_te_pgd = corr_te_pgd / float(test_num)
    print('Attacked test acc. = %f.'%acc_te_pgd)
    
 

# -------- start point
if __name__ == '__main__':
    main()