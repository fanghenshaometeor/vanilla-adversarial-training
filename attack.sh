# --------- CIFAR10-vgg/resnet ------------------
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
# model=resnet18
# model_path='./save/CIFAR10-resnet18.pth'
# model_path='./save/CIFAR10-resnet18-adv.pth'
# --------
# model=resnet20
# model_path='./save/CIFAR10-resnet20.pth'
# model_path='./save/CIFAR10-resnet20-adv.pth'
# --------
# dataset=CIFAR10
# data_dir='/media/Disk1/KunFang/data/CIFAR10/'
# -----------------------------------------------
# --------- STL10-modelA ------------------------
model=aaron
# model_path='./save/STL10-modela.pth'
model_path='./save/STL10-modela-adv.pth'
dataset=STL10
data_dir='/media/Disk1/KunFang/data/STL10/'
# -----------------------------------------------
gpu_id=1

python attack.py \
    --model ${model} \
    --model_path ${model_path} \
    --dataset ${dataset} \
    --data_dir ${data_dir}   \
    --gpu_id ${gpu_id}