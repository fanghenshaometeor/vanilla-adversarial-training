# -------- model: vgg/wideresnet/inception -----
# model=vgg11
# model=vgg13
# model=vgg16
# model=vgg19
# model=wrn28x5
# model=wrn28x10
# model=resnet20
model=resnet32
# -------- dataset: cifar10/cifar100/imagenet --
# dataset=CIFAR10
# data_dir='/media/Disk1/KunFang/data/CIFAR10/'
# ----
# dataset=CIFAR100
# data_dir='/media/Disk1/KunFang/data/CIFAR100/'
# ----
dataset=svhn
data_dir='/media/Disk1/KunFang/data/SVHN/'
# -------- enable adv. training ----------------
# adv_train=False
adv_train=True

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29503 train.py \
    --model ${model} \
    --dataset ${dataset} \
    --data_dir ${data_dir} \
    --adv_train ${adv_train}
