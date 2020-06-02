# vanilla-adversarial-training

This repo provides the PyTorch code for both **vanilla** training and **adversarial** training deep neural networks, including
- **CIFAR10 + vgg16**
- **CIFAR10 + resnet18**
- **STL10 + modelA**

## File Descriptions

`train.py` & `train.sh` : training python and shell scripts

`attack.py` & `attack.sh` : attack python and shell scripts

`attackers.py` : adversarial attack functions, including FGSM & PGD attacks

`utils.py` : help functions

`model/` : model definitions

`save/` : saved model files


## Results

### **clean sample** accuracy

model                   | training acc.(%) | test acc.(%)
:-:                     | :-:              | :-:
CIFAR10 + vgg16         | 100              | 93.18
CIFAR10 + vgg16-adv     | 100              | 88.77
CIFAR10 + resnet18      | 100              | 95.42
CIFAR10 + resnet18-adv  | 100              | 91.54
STL10 + modelA          | 100              | 77.425
STL10 + modelA-adv      | 100              | 72.9875

### **adversarial example** accuracy (%) (only test set)
   - For FGSM attack, we test the accuracies variation w.r.t. the step size $\epsilon$
   - For PGD attack, we test the accuracies variation w.r.t. the $l_\infty$ bound $\delta$, with fixed $step\ size\ \alpha=0.01,\ iterations=7$.

**1. CIFAR10 + vgg16/resnet18**

FGSM-$\epsilon$ | 1/255 | 2/255 | 3/255 | 4/255 | 5/255 | 6/255 | 7/255 | 8/255 | 9/255 | 10/255 | 11/255 | 12/255
 :-:            | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:    | :-:    | :-:
VGG16_bn        | 64.71 | 46.83 | 36.35 | 30.16 | 26.12 | 23.30 | 21.21 | 19.52 | 18.34 | 17.48  | 16.56  | 15.91
VGG16_bn-adv    | 83.46 | 76.96 | 71.03 | 66.79 | 63.38 | 60.65 | 58.62 | 56.51 | 54.61 | 52.78  | 51.02  | 49.22
ResNet18        | 69.29 | 56.99 | 50.82 | 47.02 | 43.63 | 41.13 | 38.92 | 36.95 | 34.94 | 33.03  | 31.36  | 29.69
ResNet18-adv    | 87.08 | 81.94 | 76.80 | 71.56 | 67.16 | 63.01 | 59.52 | 56.50 | 53.80 | 51.47  | 49.68  | 48.11


PGD-$\delta$ | 1/255 | 2/255 | 3/255 | 4/255 | 5/255 | 6/255 | 7/255 | 8/255 | 9/255 | 10/255 | 11/255 | 12/255
 :-:         | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:    | :-:    | :-:
VGG16_bn     | 57.44 | 19.88 | 4.40  | 1.00  | 0.23  | 0.05  | 0.06  | 0.02  | 0.03  | 0.01   | 0.01   | 0.01
VGG16_bn-adv | 83.20 | 75.27 | 66.72 | 59.23 | 53.66 | 49.90 | 46.77 | 44.23 | 42.24 | 40.86  | 39.76  | 38.95
ResNet18     | 57.66 | 19.41 | 5.37  | 1.73  | 0.63  | 0.22  | 0.12  | 0.07  | 0.03  | 0.02   | 0.05   | 0.04
ResNet18-adv | 86.94 | 80.91 | 73.56 | 65.55 | 57.59 | 50.33 | 45.03 | 40.35 | 36.94 | 34.49  | 32.49  | 31.01

**2. STL10 + modelA**

FGSM-$\epsilon$ | 1/255 | 2/255 | 3/255 | 4/255 | 5/255 | 6/255 | 7/255 | 8/255 | 9/255 | 10/255 | 11/255 | 12/255
 :-:            | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:    | :-:    | :-:
modelA          |53.9375|35.0375|23.0375|16.2875|11.8875| 9.075 | 7.1125| 5.875 | 4.975 | 4.2375 | 3.7625 | 3.2750
modelA-adv      |65.175 | 57.55 | 50.525|43.3125| 37.05 |31.9875|27.3875| 23.7  |20.1625| 17.2875| 14.95  | 12.875

PGD-$\delta$ | 1/255 | 2/255 | 3/255 | 4/255 | 5/255 | 6/255 | 7/255 | 8/255 | 9/255 | 10/255 | 11/255 | 12/255
 :-:         | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:    | :-:    | :-:
modelA       |53.1375| 30.75 |16.3625| 8.0875| 3.6375| 1.8375| 0.9   | 0.45  | 0.3   | 0.1875 | 0.1    | 0.0625
modelA-adv   |64.9375| 56.825|48.7375|40.1375|33.0875| 27.275|22.4375| 18.075| 14.675| 12.325 | 10.575 | 9.20

## Usage

### attack

We **provide 6 trained models** in `save` folder, including vanilla and adversarial training vgg16/resnet18 on CIFAR10 and [modelA](https://github.com/aaron-xichen/pytorch-playground/blob/master/stl10/model.py) on STL10.
Users can directly run the `attack.sh` shell script on command line to test the defence ability of different models.
The results should be similar with the values in the 4 tables above.
In addition, users can manually change the attack parameters in the `attackers.py` python script for more results under different settings.
```
$ sh attack.sh
```
- `model` : Please specify the target model network architecture. `vgg16`, `resnet18` or `aaron` are optional.
- `model_path` : Please specify the target model path.
- `dataset` & `data_dir` : Please specify the dataset name and path.
- `gpu_id` : GPU device index.

### training

To reproduce the provided model, users can run the `train.sh` shell scripts on command line.
```
$ sh train.sh
```
- `model` : Please specify the target model network architecture. `vgg16`, `resnet18` or `aaron` are optional.
- `dataset` & `data_dir` : Please specify the dataset name and path.
- `model_dir` : Please specify where to save the trained model.
- `log_dir` : Please specify where to save the log files.
- `gpu_id` : GPU device index.
- `adv_train` : Please specify whether to use adversarial training. `True` or `False`.

**ATTENTION** 
- The **mean-var normalization** preprocess is **removed** in both vanilla-training and adversarial-training to keep the image pixel range [0,1].
- The adversarial training is **PGD-based**, i.e., the adversarial examples in training are generated by PGD attack.
- In adversarial training, the network prameters are **updated twice** in each iteration, i.e., one normal updation on the clean samples followed by the other updation on the adversarial examples.


## Dependencies
- python 3.6 (miniconda)
- PyTorch 1.4.0

If u find the codes useful, welcome to fork and star this repo :)