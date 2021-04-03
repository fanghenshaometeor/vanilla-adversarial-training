# -------- model: vgg/wideresnet ---------------
# model=vgg11
# model=vgg13
# model=vgg16
# model=vgg19
model=wrn28x5
# model=wrn28x10
# -------- dataset: cifar10/cifar100 -----------
# dataset=CIFAR10
# data_dir='/media/Disk1/KunFang/data/CIFAR10/'
# ----
dataset=CIFAR100
data_dir='/media/Disk1/KunFang/data/CIFAR100/'
# -------- enable adv. training ----------------
adv_train=False
# adv_train=True

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 train.py \
    --model ${model} \
    --dataset ${dataset} \
    --data_dir ${data_dir} \
    --adv_train ${adv_train}
