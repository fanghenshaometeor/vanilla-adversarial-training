# --------- CIFAR10-vgg  ------------------------
# model=vgg11
# model_path='./save/CIFAR10-vgg11.pth'
# model_path='./save/CIFAR10-vgg11-adv.pth'
# --------
# model=vgg13
# model_path='./save/CIFAR10-vgg13.pth'
# model_path='./save/CIFAR10-vgg13-adv.pth'
# --------
# model=vgg16
# model_path='./save/CIFAR10-vgg16.pth'
# model_path='./save/CIFAR10-vgg16-adv.pth'
# --------
# model=vgg19
# model_path='./save/CIFAR10-vgg19.pth'
# model_path='./save/CIFAR10-vgg19-adv.pth'
# --------
dataset=CIFAR10
data_dir='/media/Disk1/KunFang/data/CIFAR10/'
# -----------------------------------------------
# --------- CIFAR100-wideresnet -----------------
# model=wrn28x5
# model_path='./save/CIFAR100-wrn28x5.pth'
# model_path='./save/CIFAR100-wrn28x5-adv.pth'
# --------
# model=wrn28x10
# model_path='./save/CIFAR100-wrn28x10.pth'
# model_path='./save/CIFAR100-wrn28x10-adv.pth'
# --------
# dataset=CIFAR100
# data_dir='/media/Disk1/KunFang/data/CIFAR100/'
# -----------------------------------------------
# --------- SVHN-resnet -------------------------
# model=resnet20
# model_path='./save/svhn/resnet20.pth'
# model_path='./save/svhn/resnet20-adv.pth'
# --------
model=resnet32
# model_path='./save/svhn/resnet32.pth'
model_path='./save/svhn/resnet32-adv.pth'
# --------
dataset=svhn
data_dir='/media/Disk1/KunFang/data/SVHN/'
# -----------------------------------------------
gpu_id=0

python attack.py \
    --model ${model} \
    --model_path ${model_path} \
    --dataset ${dataset} \
    --data_dir ${data_dir}   \
    --gpu_id ${gpu_id}