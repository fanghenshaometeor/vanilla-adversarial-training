# -------- arch: preactresnet/wideresnet --------
# arch=wrn34x10
arch=preactresnet18
# -------- dataset: cifar10/cifar100/imagenet --------
dataset=CIFAR10
data_dir='~/KunFang/data/CIFAR10/'
# ----
# dataset=CIFAR100
# data_dir='/media/Disk1/KunFang/data/CIFAR100/'
# ----
# dataset=svhn
# data_dir='/media/Disk1/KunFang/data/SVHN/'
# -------- enable adv. training ----------------
# adv_train=False
adv_train=True

CUDA_VISIBLE_DEVICES=3 python train.py \
    --arch ${arch} \
    --dataset ${dataset} \
    --data_dir ${data_dir} \
    --adv_train ${adv_train}