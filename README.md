# vanilla-adversarial-training

This repo provides the PyTorch code for both **vanilla** training and **adversarial** training deep neural networks.

## File Descriptions

`train.py,.sh` : training python and shell scripts

`attack.py,.sh` : attack python and shell scripts

`utils.py` : utility functions

`model/` : model definitions directory

<!-- ## Results

Complete results can be found in this sheet. -->


<!-- ## Usage

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
- `adv_train` : Please specify whether to use adversarial training. `True` or `False`. -->

**ATTENTION** 
- The **mean-var normalization** preprocess is included in the model definitions.
- The adversarial training is **PGD-based**: bound $l_\infty=8/255(0.031)$, step-size $2/255$ and $10$ iterations.
- In adversarial training, the network prameters are updated with adversarial examples only.
- The model is trained for $200$ epochs and the last model is selected.


## Dependencies
- python 3.6
- PyTorch 1.7.1
- AdverTorch

## 

If u find the codes useful, welcome to fork and star this repo :)