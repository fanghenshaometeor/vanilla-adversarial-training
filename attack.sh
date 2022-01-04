# --------- CIFAR10 -----------------------------
# model=resnet18
# model_path='./save/CIFAR10/resnet18/epoch200.pth'
# model_path='./save/CIFAR10/resnet18/epoch200-adv.pth'
# --------
arch=preactresnet18
# model_path='./save/CIFAR10/preactresnet18/epoch200.pth'
model_path='./save/CIFAR10/preactresnet18/epoch200-adv.pth'
# --------
# model=resnet20
# model_path='./save/CIFAR10/resnet20/epoch200.pth'
# model_path='./save/CIFAR10/resnet20/epoch200-adv.pth'
# --------
dataset=CIFAR10
data_dir='/media/Disk1/KunFang/data/CIFAR10/'
# -----------------------------------------------
# --------- CIFAR100-wideresnet -----------------
# model=wrn28x5
# model_path='./save/CIFAR100/wrn28x5.pth'
# model_path='./save/CIFAR100/wrn28x5-adv.pth'
# --------
# model=wrn28x10
# model_path='./save/CIFAR100/wrn28x10.pth'
# model_path='./save/CIFAR100/wrn28x10-adv.pth'
# --------
# dataset=CIFAR100
# data_dir='/media/Disk1/KunFang/data/CIFAR100/'
# -----------------------------------------------
attack_type=None
# attack_type=fgsm
# attack_type=pgd

CUDA_VISIBLE_DEVICES=1 python attack.py \
    --arch ${arch} \
    --model_path ${model_path} \
    --dataset ${dataset} \
    --data_dir ${data_dir} \
    --attack_type ${attack_type}