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

import os
import ast
import argparse

from utils import setup_seed
from attackers import fgsm_attack, pgd_attack

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== fix seed =============
setup_seed(666)

# ======== options ==============
parser = argparse.ArgumentParser(description='Attack Deep Neural Networks')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--model',type=str,default='vgg16',help='model architecture name')
parser.add_argument('--model_path',type=str,default='./save/CIFAR10-VGG.pth',help='saved model path')
# -------- training param. ----------
parser.add_argument('--num_classes',type=int,default=10,help='number of classes')
parser.add_argument('--batch_size',type=int,default=128,help='batch size for training (default: 256)')
parser.add_argument('--gpu_id',type=str,default='0',help='gpu device index')
args = parser.parse_args()

# ======== GPU device ========
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# -------- main function
def main():
    
    # ======== data set preprocess =============
    # ======== mean-variance normalization is removed
    if args.dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    elif args.dataset == 'CIFAR100':
        args.num_classes = 100
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform)
    elif args.dataset == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.SVHN(root=args.data_dir, split='train', download=True, 
                            transform=transform)
        testset = datasets.SVHN(root=args.data_dir, split='test', download=True, 
                            transform=transform)
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    train_num, test_num = len(trainset), len(testset)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)
    print('---- #train : %d'%train_num)
    print('---- #test  : %d'%test_num)

    # ======== load network ========
    checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
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
    elif args.model == 'resnet20':
        from model.resnet_v1 import resnet20
        net = resnet20(num_classes=args.num_classes).cuda()
    elif args.model == 'resnet32':
        from model.resnet_v1 import resnet32
        net = resnet32(num_classes=args.num_classes).cuda()
    elif args.model == 'wrn28x5':
        from model.wideresnet import wrn28
        net = wrn28(widen_factor=5, num_classes=args.num_classes).cuda()
    elif args.model == 'wrn28x10':
        from model.wideresnet import wrn28
        net = wrn28(widen_factor=10, num_classes=args.num_classes).cuda()
    else:
        assert False, "Unknow model : {}".format(args.model)
    net = nn.DataParallel(net)
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
    fgsm_epsilons = [1/255, 2/255, 3/255, 4/255, 5/255, 6/255, 7/255, 8/255, 9/255, 10/255, 11/255, 12/255]
    print('---- EPSILONS: ', fgsm_epsilons)
    for eps in fgsm_epsilons:
        print('---- current eps = %.3f...'%eps)
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