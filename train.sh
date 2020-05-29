# model=vgg16
model=resnet18
dataset=CIFAR10
data_dir='/media/Disk1/KunFang/data/CIFAR10/'
model_dir='./save/'
log_dir='./log/'
gpu_id=1

python train.py \
    --model ${model} \
    --dataset ${dataset} \
    --data_dir ${data_dir}   \
    --model_dir ${model_dir} \
    --log_dir ${log_dir} \
    --gpu_id ${gpu_id}