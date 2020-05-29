model=vgg16
# model=resnet18
dataset=CIFAR10
data_dir='/media/Disk1/KunFang/data/CIFAR10/'
model_path='./save/CIFAR10-VGG.pth'
# model_path='./save/CIFAR10-VGG-adv.pth'
# model_path='./save/CIFAR10-ResNet18.pth'
# model_path='./save/CIFAR10-Resnet18-adv.pth'
gpu_id=1

python attack.py \
    --model ${model} \
    --dataset ${dataset} \
    --data_dir ${data_dir}   \
    --model_path ${model_path} \
    --gpu_id ${gpu_id}