# vanilla-adversarial-training

This repo provides the code for both **vanilla** training and **adversarial** training **VGG/ResNet** models on CIFAR10 in PyTorch.

## File Descriptions

`vgg.py` & `ResNet.py`: model definitions

`trainCIFAR10VGG.py` & `trainCIFAR10ResNet.py`: **vanilla training** VGG or ResNet models from scratch (no defence)

`advTrainCIFAR10VGG.py` & `advTrainCIFAR10ResNet.py`: **adversarial training** VGG or ResNet models from scratch (**PGD-based**)

`attackVGG.py` & `attackResNet.py`: attack the trained model using **FGSM** attack and **PGD** attack


## Results

1. **clean sample** accuracy

model       | training acc.(%) | test acc.(%)
:-:         | :-:              | :-:
VGG16_bn    | 99.994           | 93.05
VGG16_bn-adv| 100              | 92.76
ResNet18    | 100              | 95.37
ResNet18-adv| 100              | 94.78

2. **adversarial example** accuracy (%) (only test set)
   - PGD attack settings: $bound\ \delta=0.031,\ step\ size\ \alpha=0.01,\ iterations=7$

FGSM-$\epsilon$ | 0   | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | PGD  
 :-:            |:-:  | :-:  | :-: | :-:  | :-: | :-:  | :-: | :-:
VGG16_bn        |93.05|41.23 |26.43|20.90 |18.09|16.22 |15.05|29.81
VGG16_bn-adv    |92.67|68.01 |51.47|39.60 |32.06|27.14 |24.75|71.62
ResNet18        |95.37|53.92 |43.63|37.59 |32.50|28.18 |24.75|25.27
ResNet18-adv    |94.78|72.21 |56.90|48.67 |43.66|40.74 |38.94|77.50

## Usage

We **provide 4 trained models** in `model` folder, including vanilla and adversarial training VGG and ResNet models, named as `CIFAR10-VGG.pth`, `CIFAR10-VGG-adv.pth`, `CIFAR10-ResNet18.pth` and `CIFAR10-ResNet18-adv.pth` respectively.
Users can directly run the 2 attack scripts on command line to test the defence ability of models.
The results should be similar with the values in the two tables above.
Users should **specify** the target model in attack scripts by modifying the **args.model_name** parameter.
```
python attackVGG.py
python attackResNet.py
```

To reproduce the provided model, users can run the 4 training scripts on command line.
**Pay attention to** the data folder and log folder parameters in training scripts and make appropriate modifications.
```
python trainCIFAR10VGG.py
python advTrainCIFAR10VGG.py
python trainCIFAR10ResNet.py
python advTrainCIFAR10ResNet.py
```

Welcome to fork and star this repo :)