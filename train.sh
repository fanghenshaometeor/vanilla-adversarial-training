# --------- CIFAR10-vgg16/resnet18 --------------
# model=vgg11
# model=vgg13
# model=vgg16
# model=vgg19
# model=resnet18
# model=resnet20
# dataset=CIFAR10
# data_dir='/media/Disk1/KunFang/data/CIFAR10/'
# -----------------------------------------------
# --------- CIFAR100-vgg16 ----------------------
# model=vgg16
model=resnet20
dataset=CIFAR100
data_dir='/media/Disk1/KunFang/data/CIFAR100/'
# --------- STL10-modelA ------------------------
# model=modela
# dataset=STL10
# data_dir='/media/Disk1/KunFang/data/STL10/'
# -----------------------------------------------
model_dir='./save/'
gpu_id=3
# -----------------------------------------------
# adv_train=False
adv_train=True

python train.py \
    --model ${model} \
    --dataset ${dataset} \
    --data_dir ${data_dir} \
    --model_dir ${model_dir} \
    --gpu_id ${gpu_id} \
    --adv_train ${adv_train}
