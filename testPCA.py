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

import ast
import argparse

import numpy as np
from sklearn.decomposition import PCA

from utils import *
from attackers import * 

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== fix seed =============
setup_seed(666)

# ======== options ==============
parser = argparse.ArgumentParser(description='Perform PCA on features')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--model_dir',type=str,default='./save/',help='file path for saving model')
parser.add_argument('--log_dir',type=str,default='./log/',help='file path for saving log')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--model',type=str,default='vgg16',help='model architecture name')
parser.add_argument('--model_path',type=str,default='./save/CIFAR10-VGG.pth',help='saved model path')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=256,help='batch size for training (default: 256)')
parser.add_argument('--gpu_id',type=str,default='0',help='gpu device index')
# -------- sample size for PCA --------
parser.add_argument('--sample_size',type=int,default=1000,help='sample size for PCA')
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
    elif args.dataset == 'STL10':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.STL10(root=args.data_dir, split='train', transform=transform, download=True)
        testset = datasets.STL10(root=args.data_dir, split='test', transform=transform, download=True)
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
    if args.model == 'vgg16':
        from model.vgg import vgg16_bn
        net = vgg16_bn().cuda()
    elif args.model == 'resnet18':
        from model.resnet import ResNet18
        net = ResNet18().cuda()
    elif args.model == 'aaron':
        from model.aaron import Aaron
        net = Aaron().cuda()
    else:
        assert False, "Unknow model : {}".format(args.model)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    print('-------- MODEL INFORMATION --------')
    print('---- model:      '+args.model)
    print('---- saved path: '+args.model_path)

    print('-------- START TESTING --------')
    # corr_tr, corr_te = test(net, trainloader, testloader)
    # acc_tr, acc_te = corr_tr / train_num, corr_te / test_num
    # print('Train acc. = %f; Test acc. = %f.' % (acc_tr, acc_te))

    print('-------- PCA REDUCE DIM. on clean samples & adversarial examples --------')
    print('---- sample size   = %d'%args.sample_size)
    print('---- original dim. = %d'%(512))
    print('---- reduced  dim. = %d'%(2))
    print('---- epsilon/bound in FGSM/PGD attack = %d/%d'%(8,255))
    tr_sampler = data.sampler.RandomSampler(data_source=trainset, replacement=True, num_samples=args.sample_size)
    te_sampler = data.sampler.RandomSampler(data_source=testset, replacement=True, num_samples=args.sample_size)
    batch_trainsampler = data.sampler.BatchSampler(sampler=tr_sampler, batch_size=args.sample_size, drop_last=False)
    batch_testsampler = data.sampler.BatchSampler(sampler=te_sampler, batch_size=args.sample_size, drop_last=False)
    batch_trainloader = data.DataLoader(dataset=trainset, batch_sampler=batch_trainsampler, shuffle=False)
    batch_testloader = data.DataLoader(dataset=testset, batch_sampler=batch_testsampler, shuffle=False)

    PCADecomp(net, batch_trainloader, batch_testloader, 8/255)

    return

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
            
            _, logits = net(images)

            """ print several logits """
            # print(logits[0,:])
            # print(logits[1,:])
            # print(logits[2,:])
            # print(logits[3,:])
            # print(logits[4,:])
            """ """

            logits = logits.detach()
            _, predicted = torch.max(logits.data, 1)
            
            correct_test += (predicted == labels).sum().item()
        
        for train in trainloader:
            images, labels = train
            images, labels = images.cuda(), labels.cuda()
            
            _, logits = net(images)
            logits = logits.detach()
            _, predicted = torch.max(logits.data, 1)
            
            correct_train += (predicted == labels).sum().item()  
    
    return correct_train, correct_test

# -------- PCA ---------------------
def PCADecomp(net, batch_trainloader, batch_testloader, epsilon):
    
    net.eval()

    pca_estimator = PCA(n_components=2)

    # one batch only
    for test in batch_testloader:
        images, labels = test
        images, labels = images.cuda(), labels.cuda()

        # features of clean sample
        clean_features, _ = net(images)
        clean_features = clean_features.detach().cpu().numpy()
        pca_clean_f = pca_estimator.fit_transform(clean_features)
        pca_c_test = np.hstack((pca_clean_f, labels.unsqueeze(1).cpu().numpy()))

        # features of adversarial examples w.r.t. FGSM
        perturbed_image, _ = fgsm_attack(net, images, labels, epsilon)
        adv_features, logits = net(perturbed_image)
        adv_features = adv_features.detach().cpu().numpy()
        pca_adv_f = pca_estimator.fit_transform(adv_features)
        pca_a_test_fgsm = np.hstack((pca_adv_f, labels.unsqueeze(1).cpu().numpy()))

        # features of adversarial examples w.r.t. PGD
        # perturbed_image_pgd, _ = pgd_attack(net, images, labels, epsilon)
        # adv_features, _ = net(perturbed_image_pgd)
        # adv_features = adv_features.detach().cpu().numpy()
        # pca_adv_f = pca_estimator.fit_transform(adv_features)
        # pca_a_test_pgd = np.hstack((pca_adv_f, labels.unsqueeze(1).cpu().numpy()))

        np.save(args.log_dir+'pca_test', pca_c_test)
        np.save(args.log_dir+'pca_test_FGSM', pca_a_test_fgsm)
        # np.save(args.log_dir+'pca_test_PGD', pca_a_test_pgd)


    # one batch only
    for train in batch_trainloader:
        images, labels = train
        images, labels = images.cuda(), labels.cuda()

        # features of clean sample
        clean_features, _ = net(images)
        clean_features = clean_features.detach().cpu().numpy()
        pca_clean_f = pca_estimator.fit_transform(clean_features)
        pca_c_train = np.hstack((pca_clean_f, labels.unsqueeze(1).cpu().numpy()))

        # features of adversarial examples w.r.t. FGSM
        perturbed_image, _ = fgsm_attack(net, images, labels, epsilon)
        adv_features, _ = net(perturbed_image)
        adv_features = adv_features.detach().cpu().numpy()
        pca_adv_f = pca_estimator.fit_transform(adv_features)
        pca_a_train_fgsm = np.hstack((pca_adv_f, labels.unsqueeze(1).cpu().numpy())) 

        # features of adversarial examples w.r.t. PGD
        # perturbed_image_pgd, _ = pgd_attack(net, images, labels, epsilon)
        # adv_features, _ = net(perturbed_image_pgd)
        # adv_features = adv_features.detach().cpu().numpy()
        # pca_adv_f = pca_estimator.fit_transform(adv_features)
        # pca_a_train_pgd = np.hstack((pca_adv_f, labels.unsqueeze(1).cpu().numpy()))        

        np.save(args.log_dir+'pca_train', pca_c_train)
        np.save(args.log_dir+'pca_train_FGSM', pca_a_train_fgsm)
        # np.save(args.log_dir+'pca_train_PGD', pca_a_train_pgd)
    
    return

# -------- start point
if __name__ == '__main__':
    main()