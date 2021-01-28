# vanilla-adversarial-training

This repo provides the PyTorch code for both **vanilla** training and **adversarial** training deep neural networks, including
- **CIFAR10 + vgg11/vgg13/vgg16/vgg19/resnet18/resnet20**
- **CIFAR100 + vgg16/resnet20**
- **STL10 + [modelA](https://github.com/aaron-xichen/pytorch-playground/blob/master/stl10/model.py)**

## File Descriptions

`train.py,.sh` : training python and shell scripts

`attack.py,.sh` : attack python and shell scripts

`model/` : model definitions directory

`save/` : saved model files directory


## Results

### **clean sample** accuracy

dataset + model         | training acc.(%) | test acc.(%)
:-:                     | :-:              | :-:
CIFAR10 + vgg11         | 99.992           | 91.33
CIFAR10 + vgg11-adv     | 100.00           | 87.04
CIFAR10 + vgg13         | 99.998           | 93.19
CIFAR10 + vgg13-adv     | 100.00           | 89.23
CIFAR10 + vgg16         | 100.00           | 93.18
CIFAR10 + vgg16-adv     | 100.00           | 88.77
CIFAR10 + vgg19         | 99.998           | 92.86
CIFAR10 + vgg19-adv     | 100.00           | 89.07
CIFAR10 + resnet18      | 100.00           | 95.42
CIFAR10 + resnet18-adv  | 100.00           | 91.54
CIFAR10 + resnet20      | 99.964           | 91.71
CIFAR10 + resnet20-adv  | 97.586           | 88.03
CIFAR100 + vgg16        | 99.97            | 72.45
CIFAR100 + vgg16-adv    | 99.96            | 62.62
CIFAR100 + resnet20     | 93.926           | 67.31
CIFAR100 + resnet20-adv | 76.872           | 61.94
STL10 + modelA          | 100.00           | 77.425
STL10 + modelA-adv      | 100.00           | 72.9875

### **adversarial example** accuracy (%) (only test set)
   - For FGSM attack, we test the accuracies variation w.r.t. the step size $\epsilon$
   - For PGD attack, we test the accuracies variation w.r.t. the $l_\infty$ bound $\delta$, with fixed $step\ size\ \alpha=0.01,\ iterations=7$.

**1. CIFAR10 + vgg/resnet

FGSM-$\epsilon$ (**/255**) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12
 :-:         | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:    | :-:    | :-:
vgg11        | 69.35 | 52.80 | 42.23 | 34.68 | 29.40 | 25.22 | 22.31 | 19.59 | 17.89 | 16.44  | 15.27  | 14.43
vgg11-adv    | 80.92 | 74.14 | 67.13 | 60.48 | 54.76 | 49.68 | 45.34 | 41.52 | 38.34 | 35.34  | 32.90  | 30.52
vgg13        | 65.53 | 50.53 | 41.94 | 36.88 | 33.00 | 29.79 | 27.58 | 25.93 | 24.25 | 22.77  | 21.20  | 20.10
vgg13-adv    | 83.37 | 77.16 | 71.23 | 65.65 | 60.72 | 56.32 | 52.20 | 48.67 | 45.94 | 43.31  | 41.11  | 39.14
vgg16        | 64.71 | 46.83 | 36.35 | 30.16 | 26.12 | 23.30 | 21.21 | 19.52 | 18.34 | 17.48  | 16.56  | 15.91
vgg16-adv    | 83.46 | 76.96 | 71.03 | 66.79 | 63.38 | 60.65 | 58.62 | 56.51 | 54.61 | 52.78  | 51.02  | 49.22
vgg19        | 62.89 | 40.82 | 28.41 | 21.15 | 16.51 | 13.61 | 11.65 | 10.35 | 9.52  | 9.08   | 8.67   | 8.42
vgg19-adv    | 83.05 | 76.94 | 71.25 | 67.06 | 63.47 | 60.46 | 57.63 | 55.25 | 53.21 | 51.03  | 48.88  | 47.03
resnet18     | 69.29 | 56.99 | 50.82 | 47.02 | 43.63 | 41.13 | 38.92 | 36.95 | 34.94 | 33.03  | 31.36  | 29.69
resnet18-adv | 87.08 | 81.94 | 76.80 | 71.56 | 67.16 | 63.01 | 59.52 | 56.50 | 53.80 | 51.47  | 49.68  | 48.11
resnet20     | 42.07 | 26.11 | 20.24 | 17.60 | 15.79 | 14.44 | 13.96 | 13.36 | 13.16 | 13.07  | 12.82  | 12.55
resnet20-adv | 81.78 | 74.94 | 67.59 | 60.25 | 52.85 | 46.17 | 40.28 | 35.06 | 30.14 | 26.40  | 23.17  | 20.41


PGD-$\delta$ (**/255**) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12
 :-:         | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:    | :-:    | :-:
vgg11        | 65.35 | 32.92 | 12.82 | 5.09  | 2.08  | 1.09  | 0.54  | 0.36  | 0.22  | 0.12   | 0.08   | 0.07
vgg11-adv    | 80.67 | 72.54 | 62.75 | 53.46 | 44.22 | 36.47 | 30.24 | 25.81 | 22.26 | 19.71  | 17.86  | 16.29
vgg13        | 57.65 | 21.18 | 5.86  | 1.84  | 0.71  | 0.35  | 0.16  | 0.09  | 0.06  | 0.04   | 0.07   | 0.02
vgg13-adv    | 83.10 | 75.77 | 67.48 | 58.38 | 49.28 | 42.34 | 37.02 | 32.48 | 29.05 | 26.19  | 24.35  | 22.68
vgg16        | 57.44 | 19.88 | 4.40  | 1.00  | 0.23  | 0.05  | 0.06  | 0.02  | 0.03  | 0.01   | 0.01   | 0.01
vgg16-adv    | 83.20 | 75.27 | 66.72 | 59.23 | 53.66 | 49.90 | 46.77 | 44.23 | 42.24 | 40.86  | 39.76  | 38.95
vgg19        | 57.42 | 19.16 | 3.67  | 0.76  | 0.17  | 0.09  | 0.02  | 0.00  | 0.01  | 0.01   | 0.00   | 0.00
vgg19-adv    | 82.78 | 75.23 | 66.78 | 58.06 | 51.36 | 45.97 | 42.18 | 39.30 | 36.42 | 34.13  | 32.63  | 31.30
resnet18     | 57.74 | 19.48 | 5.58  | 1.85  | 0.52  | 0.21  | 0.18  | 0.08  | 0.03  | 0.03   | 0.03   | 0.03
resnet18-adv | 86.94 | 80.94 | 73.53 | 65.53 | 57.51 | 50.40 | 44.96 | 40.42 | 37.04 | 34.36  | 32.44  | 30.74
resnet20     | 37.68 | 5.98  | 0.46  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00   | 0.00   | 0.00
resnet20-adv | 81.62 | 73.67 | 64.14 | 54.08 | 43.86 | 35.50 | 28.33 | 22.26 | 17.66 | 14.28  | 11.62  | 9.83

**2. CIFAR100 + vgg/resnet**

FGSM-$\epsilon$ | 1/255 | 2/255 | 3/255 | 4/255 | 5/255 | 6/255 | 7/255 | 8/255 | 9/255 | 10/255 | 11/255 | 12/255
 :-:         | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:    | :-:    | :-:
vgg16        | 36.11 | 27.90 | 24.56 | 22.39 | 20.49 | 18.88 | 17.26 | 16.09 | 14.88 | 13.82  | 12.79  | 11.71
vgg16-adv    | 54.13 | 46.73 | 40.67 | 35.47 | 31.36 | 27.91 | 25.15 | 22.80 | 20.88 | 19.22  | 17.95  | 16.80
resnet20     | 12.64 | 7.60  | 6.75  | 6.19  | 5.78  | 5.55  | 5.54  | 5.36  | 5.10  | 4.83   | 4.64   | 4.63
resnet20-adv | 53.00 | 44.12 | 37.21 | 31.08 | 25.58 | 21.26 | 17.86 | 14.94 | 12.51 | 10.34  | 8.87   | 7.74

PGD-$\delta$ | 1/255 | 2/255 | 3/255 | 4/255 | 5/255 | 6/255 | 7/255 | 8/255 | 9/255 | 10/255 | 11/255 | 12/255
 :-:         | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:    | :-:    | :-:
vgg16        | 28.56 | 9.81  | 3.94  | 1.71  | 0.94  | 0.47  | 0.32  | 0.18  | 0.15  | 0.06   | 0.12   | 0.03
vgg16-adv    | 53.67 | 45.02 | 36.87 | 30.08 | 24.04 | 19.60 | 16.26 | 13.90 | 11.92 | 10.57  | 9.71   | 8.98
resnet20     | 11.15 | 1.63  | 0.13  | 0.01  | 0.03  | 0.01  | 0.01  | 0.00  | 0.00  | 0.00   | 0.00   | 0.00
resnet20-adv | 52.84 | 42.93 | 34.23 | 25.93 | 19.80 | 14.79 | 11.15 | 8.39  | 6.36  | 5.09   | 4.12   | 3.35

**3. STL10 + modelA**

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

We **provide trained models** in `save` folder, including vanilla and adversarial training vgg/resnet on CIFAR10 and [modelA](https://github.com/aaron-xichen/pytorch-playground/blob/master/stl10/model.py) on STL10.
Users can directly run the `attack.sh` shell script on command line to check the robustness of these models.
The results should be similar with the values in the 5 tables above.
In addition, users can manually change the attack parameters in the `attackers.py` python script for more results under different settings.
```
$ sh attack.sh
```
- `model` : Please specify the target model network architecture.
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
- `gpu_id` : GPU device index.
- `adv_train` : Please specify whether to use adversarial training. `True` or `False`.

**ATTENTION** 
- The **mean-var normalization** preprocess is **removed** in both vanilla-training and adversarial-training to keep the image pixel range [0,1].
- The adversarial training is **PGD-based**, i.e., the adversarial examples in training are generated by PGD attack.
- In adversarial training, the network prameters are **updated twice** in each iteration, i.e., one normal updation on the clean samples followed by the other updation on the adversarial examples.


## Dependencies
- python 3.6 (miniconda)
- PyTorch 1.5.0

## 

If u find the codes useful, welcome to fork and star this repo :)