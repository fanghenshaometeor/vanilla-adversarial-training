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
from attackers import *
from model.vgg import *
from model.resnet import *

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== fix seed =============
setup_seed(666)

# ======== options ==============
parser = argparse.ArgumentParser(description='Attack Deep Neural Networks on CIFAR10')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--model_dir',type=str,default='./save/',help='file path for saving model')
parser.add_argument('--log_dir',type=str,default='./log/',help='file path for saving log')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--model',type=str,default='vgg16',help='model name')
parser.add_argument('--model_path',type=str,default='./save/CIFAR10-VGG.pth',help='model path')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=256,help='input batch size for training (default: 512)')
parser.add_argument('--gpu_id',type=str,default='0',help='gpu device index')
args = parser.parse_args()

# ======== GPU device ========
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# -------- main function
def main():
    
    # ======== load CIFAR10 data set 32 x 32  =============
    if args.dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    else:
        print('UNSUPPORTED DATASET '+args.dataset)
        return
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    train_num, test_num = len(trainset), len(testset)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)
    print('---- #train : %d'%train_num)
    print('---- #test  : %d'%test_num)

    # ======== load network ========
    checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
    if args.model == 'vgg16':
        net = vgg16_bn().cuda()
    elif args.model == 'resnet18':
        net = ResNet18().cuda()
    else:
        print('UNSUPPORTED MODEL '+args.model)
        return
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    print('-------- MODEL INFORMATION --------')
    print('---- model:      '+args.model)
    print('---- saved path: '+args.model_path)

    print('-------- START TESTING --------')
    corr_tr, corr_te = test(net, trainloader, testloader)
    acc_tr, acc_te = corr_tr / train_num, corr_te / test_num
    print('Train acc. = %f; Test acc. = %f.' % (acc_tr, acc_te))

    print('-------- START FGSM ATTACK --------')
    fgsm_epsilons = [.05, .1, .15, .2, .25, .3, .35, .4]
    print('---- EPSILONS: ', fgsm_epsilons)
    for eps in fgsm_epsilons:
        print('---- current eps = %.2f...'%eps)
        correct_te_fgsm = attack(net, testloader, eps, "FGSM")
        acc_te_fgsm = correct_te_fgsm / float(test_num)
        print('Attacked test acc. = %f.'%acc_te_fgsm)

    print('-------- START PGD ATTACK -------')
    pgd_epsilons = [1/255, 2/255, 3/255, 4/255, 5/255, 6/255, 7/255, 8/255, 9/255, 10/255, 11/255, 12/255]
    print('---- EPSILON: ', pgd_epsilons)
    for eps in pgd_epsilons:
        print('---- current eps = %.3f...'%eps)
        corr_te_pgd = attack(net, testloader, eps, "PGD")
        acc_te_pgd = corr_te_pgd / float(test_num)
        print('Attacked test acc. = %f.'%acc_te_pgd)

# -------- test model ---------------
def test(net, trainloader, testloader):
    
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


# -------- attack model --------
def attack(net, testloader, epsilon, attackType):

    correct = 0

    net.eval()

    # loop all examples in test set
    for test in testloader:
        image, label = test
        image, label = image.cuda(), label.cuda()

        # generate adversarial examples
        if attackType == "FGSM":
            perturbed_image = fgsm_attack(net, image, label, epsilon)
        elif attackType == "PGD":
            perturbed_image = pgd_attack(net, image, label, epsilon)

        # re-classify
        logits = net(perturbed_image)

        _, final_pred = torch.max(logits.data, 1)
        correct = correct + (final_pred == label).sum().item() 
    return correct


# -------- start point
if __name__ == '__main__':
    main()