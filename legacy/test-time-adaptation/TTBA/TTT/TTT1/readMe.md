# Test-Time Training with Self-Supervision for Generalization under Distribution Shifts
[Paper link](https://proceedings.mlr.press/v119/sun20b.html)

## Softerware Environment
### Original environment
The original paper's software version description is very unclear.
+ Pytorch 1.0+
+ Python 3.7
+ tqdm (unknown)
+ colorama (unknown)

Self defined original environment:
```shell
conda create --name my python=3.9
conda activate my
# cuda 11.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install tqdm==4.66.4
conda install colorama==0.4.6
conda install matplotlib==3.8.4
```shell

## Dataset
[CIFAR-10-C](https://zenodo.org/records/2535967#.Xaf8uedKj-Y)<br/>
[CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1/tree/master)

## Code Reference
[TTT](https://github.com/yueatsprograms/ttt_cifar_release/tree/master)
