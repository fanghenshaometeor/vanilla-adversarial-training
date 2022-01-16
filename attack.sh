# -----------------------------------------------
arch=preactresnet18
# model_path='./save/CIFAR10/preactresnet18-adv/epoch100.pth'
model_path='./save/CIFAR10/preactresnet18-adv/best.pth'
# --------
dataset=CIFAR10
data_dir='~/KunFang/data/CIFAR10/'
# --------
# dataset=CIFAR100
# data_dir='/media/Disk1/KunFang/data/CIFAR100/'
# -----------------------------------------------
attack_type=None
# attack_type=fgsm
# attack_type=pgd

CUDA_VISIBLE_DEVICES=0 python attack.py \
    --arch ${arch} \
    --model_path ${model_path} \
    --dataset ${dataset} \
    --data_dir ${data_dir} \
    --attack_type ${attack_type}