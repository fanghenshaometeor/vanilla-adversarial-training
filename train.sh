# ------- CIFAR10 + vgg16 / resnet18
# model=vgg16
# model=resnet18
# dataset=CIFAR10
# data_dir='/media/Disk1/KunFang/data/CIFAR10/'
# ------- STL10 + modelA
model=aaron
dataset=STL10
data_dir='/media/Disk1/KunFang/data/STL10/'
# -------
model_dir='./save/'
log_dir='./log/'
gpu_id=2
# adv_train=False
adv_train=True

python train.py \
    --model ${model} \
    --dataset ${dataset} \
    --data_dir ${data_dir}   \
    --model_dir ${model_dir} \
    --log_dir ${log_dir} \
    --gpu_id ${gpu_id} \
    --adv_train ${adv_train}
