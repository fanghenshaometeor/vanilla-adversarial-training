# -------- arch: vgg/wideresnet/inception -----
# arch=vgg11
# arch=vgg13
# arch=vgg16
# arch=vgg19
# arch=wrn28x5
# arch=wrn28x10
# arch=resnet20
# arch=resnet32
arch=preactresnet18
# -------- dataset: cifar10/cifar100/imagenet --
dataset=CIFAR10
data_dir='/media/Disk1/KunFang/data/CIFAR10/'
# ----
# dataset=CIFAR100
# data_dir='/media/Disk1/KunFang/data/CIFAR100/'
# ----
# dataset=svhn
# data_dir='/media/Disk1/KunFang/data/SVHN/'
# -------- enable adv. training ----------------
adv_train=False
# adv_train=True

CUDA_VISIBLE_DEVICES=0 python train.py \
    --arch ${arch} \
    --dataset ${dataset} \
    --data_dir ${data_dir} \
    --adv_train ${adv_train}