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

from utils import *
from attackers import *

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== fix seed =============
setup_seed(666)

# ======== options ==============
parser = argparse.ArgumentParser(description='Attack Deep Neural Networks')
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
# -------- save adv. images --------
parser.add_argument('--save_adv_img',type=ast.literal_eval,dest='save_adv_img',help='save adversarial examples')
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

    """ change the parameters (x100) in the last linear layer """
    # print("-------- before change --------")
    # for name, param in net.named_parameters():
    #     # print(name, param.size())
    #     if name == 'classifier.0.weight':
    #         print(param)
    #         print("weight param. l_\infty norm = %f"%torch.norm(param,p=float('inf')))
    #         print("weight param. l_1      norm = %f"%torch.norm(param,p=1))
    #         print("weight param. l_2      norm = %f"%torch.norm(param,p=2))
    #         checkpoint['state_dict'][name] = param * 100
    #     if name == 'classifier.0.bias':
    #         print(param)
    #         print("bias param. l_\infty norm = %f"%torch.norm(param,p=float('inf')))
    #         print("bias param. l_1      norm = %f"%torch.norm(param,p=1))
    #         print("bias param. l_2      norm = %f"%torch.norm(param,p=2))
    #         checkpoint['state_dict'][name] = param * 100
    # net.load_state_dict(checkpoint['state_dict'])
    # net.eval()
    # print("-------- after change --------")
    # for name, param in net.named_parameters():
    #     if name == 'classifier.0.weight':
    #         print(param)
    #         print("weight param. l_\infty norm = %f"%torch.norm(param,p=float('inf')))
    #         print("weight param. l_1      norm = %f"%torch.norm(param,p=1))
    #         print("weight param. l_2      norm = %f"%torch.norm(param,p=2))
    #     if name == 'classifier.0.bias':
    #         print(param)
    #         print("bias param. l_\infty norm = %f"%torch.norm(param,p=float('inf')))
    #         print("bias param. l_1      norm = %f"%torch.norm(param,p=1))
    #         print("bias param. l_2      norm = %f"%torch.norm(param,p=2))

    """ """

    print('-------- START TESTING --------')
    corr_tr, corr_te = test(net, trainloader, testloader)
    acc_tr, acc_te = corr_tr / train_num, corr_te / test_num
    print('Train acc. = %f; Test acc. = %f.' % (acc_tr, acc_te))

    # return

    print('-------- START FGSM ATTACK --------')
    fgsm_epsilons = [1/255, 2/255, 3/255, 4/255, 5/255, 6/255, 7/255, 8/255, 9/255, 10/255, 11/255, 12/255]
    print('---- EPSILONS: ', fgsm_epsilons)
    for eps in fgsm_epsilons:
        print('---- current eps = %.3f...'%eps)
        corr_te_fgsm, avg_perturbation_fgsm, avg_grad_sign_fgsm = attack(net, testloader, eps, "FGSM")
        acc_te_fgsm = corr_te_fgsm / float(test_num)
        avg_perturbation_fgsm["linf"] = avg_perturbation_fgsm["linf"] / float(test_num)
        avg_perturbation_fgsm["l1"] = avg_perturbation_fgsm["l1"] / float(test_num)
        avg_perturbation_fgsm["l2"] = avg_perturbation_fgsm["l2"] / float(test_num)
        avg_grad_sign_fgsm = avg_grad_sign_fgsm / float(test_num)
        print('Attacked test acc. = %f.'%acc_te_fgsm)
        print('Average perturbation of linf = %f'%avg_perturbation_fgsm["linf"])
        print('Average perturbation of l1   = %f'%avg_perturbation_fgsm["l1"])
        print('Average perturbation of l2   = %f'%avg_perturbation_fgsm["l2"])
        print('Average gradient sign sum = %f'%avg_grad_sign_fgsm)

    print('-------- START PGD ATTACK -------')
    pgd_epsilons = [1/255, 2/255, 3/255, 4/255, 5/255, 6/255, 7/255, 8/255, 9/255, 10/255, 11/255, 12/255]
    print('---- EPSILON: ', pgd_epsilons)
    for eps in pgd_epsilons:
        print('---- current eps = %.3f...'%eps)
        corr_te_pgd, avg_perturbation_pgd, avg_grad_sign_pgd = attack(net, testloader, eps, "PGD")
        acc_te_pgd = corr_te_pgd / float(test_num)
        avg_perturbation_pgd["linf"] = avg_perturbation_pgd["linf"] / float(test_num)
        avg_perturbation_pgd["l1"] = avg_perturbation_pgd["l1"] / float(test_num)
        avg_perturbation_pgd["l2"] = avg_perturbation_pgd["l2"] / float(test_num)
        avg_grad_sign_pgd = avg_grad_sign_pgd / float(test_num)
        print('Attacked test acc. = %f.'%acc_te_pgd)
        print('Average perturbation of linf = %f'%avg_perturbation_pgd["linf"])
        print('Average perturbation of l1   = %f'%avg_perturbation_pgd["l1"])
        print('Average perturbation of l2   = %f'%avg_perturbation_pgd["l2"])
        print('Average gradient sign sum = %f'%avg_grad_sign_pgd)
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


# -------- attack model --------
def attack(net, testloader, epsilon, attackType):

    correct = 0

    net.eval()
    
    # initialization average perturbation
    avg_perturbation = {}
    avg_perturbation["linf"] = 0
    avg_perturbation["l1"] = 0
    avg_perturbation["l2"] = 0

    # initialization average gradient sign
    avg_grad_sign = 0

    for test in testloader:
        image, label = test
        image, label = image.cuda(), label.cuda()

        # generate adversarial examples
        if attackType == "FGSM":
            perturbed_image, batch_grad_sign_sum = fgsm_attack(net, image, label, epsilon)
        elif attackType == "PGD":
            perturbed_image, batch_grad_sign_sum = pgd_attack(net, image, label, epsilon)
        
        """ compute average perturbation """
        # print((perturbed_image-image).size())
        # print(torch.norm((perturbed_image-image).detach().cpu(),p=float('inf'),dim=(1,2,3)).size())
        avg_perturbation["linf"] = avg_perturbation["linf"] + torch.norm((perturbed_image-image).detach().cpu(), p=float('inf'), dim=(1,2,3)).sum()
        avg_perturbation["l1"]   = avg_perturbation["l1"]   + torch.norm((perturbed_image-image).detach().cpu(), p=1, dim=(1,2,3)).sum()
        avg_perturbation["l2"]   = avg_perturbation["l2"]   + torch.norm((perturbed_image-image).detach().cpu(), p=2, dim=(1,2,3)).sum()
        """ """

        """ compute average gradient sign """
        avg_grad_sign = avg_grad_sign + batch_grad_sign_sum
        """ """

        if args.save_adv_img:
            adv_examples = perturbed_image[0:5,:,:,:].squeeze().detach().cpu().numpy()
            np.save('./log/%s-%.3f-vgg-adv'%(attackType,epsilon), adv_examples)

        # re-classify
        _, logits = net(perturbed_image)

        _, final_pred = torch.max(logits.data, 1)
        correct = correct + (final_pred == label).sum().item() 
    return correct, avg_perturbation, avg_grad_sign


# -------- start point
if __name__ == '__main__':
    main()