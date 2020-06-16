# --------- CIFAR10-vgg16/resnet18 --------------
model=vgg16
# model=resnet18
# model_path='./save/CIFAR10-VGG.pth'
model_path='./save/CIFAR10-VGG-adv.pth'
# model_path='./save/CIFAR10-ResNet18.pth'
# model_path='./save/CIFAR10-Resnet18-adv.pth'
dataset=CIFAR10
data_dir='/media/Disk1/KunFang/data/CIFAR10/'
# -----------------------------------------------
# --------- STL10-modelA ------------------------
# model=aaron
# model_path='./save/STL10-aaron.pth'
# model_path='./save/STL10-aaron-adv.pth'
# dataset=STL10
# data_dir='/media/Disk1/KunFang/data/STL10/'
# -----------------------------------------------
gpu_id=3
save_adv_img=False

python attack.py \
    --model ${model} \
    --model_path ${model_path} \
    --dataset ${dataset} \
    --data_dir ${data_dir}   \
    --gpu_id ${gpu_id} \
    --save_adv_img ${save_adv_img}