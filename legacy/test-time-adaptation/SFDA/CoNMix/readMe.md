# CoNMix for Source-free Single and Multi-target Domain Adaptation
[Origin Link](https://sites.google.com/view/conmix-vcl)


## Introduction
+ Source-free Single and Multi-target Domain Adaptation
+ Pseudo labels
+ Consistency Training


## Running Environment
### Origin Paper
```shell
conda create --name 'my' python=3.8 -y
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install -y -c anaconda scipy==1.7.3
conda install -y -c anaconda scikit-learn==1.2.1
conda install -y -c anaconda seaborn==0.11.2
conda install -y matplotlib==3.6.2
pip install ml-collections==0.1.1
pip install tqdm==4.64.1
pip install wandb==0.17.1
pip install timm==0.6.12
```
```text
datasets==2.9.0
matplotlib==3.6.2
medpy==0.4.0
ml_collections==0.1.1
numpy==1.23.5
pandas==1.5.2
Pillow==9.4.0
scikit_learn==1.2.1
scipy==1.7.3
seaborn==0.11.2
SimpleITK==2.2.1
tensorboardX==2.5.1
timm==0.6.12
torch==1.13.1
torchvision==0.14.1
tqdm==4.64.1
```
### This Project
```shell
conda create --name 'my' python=3.9 -y
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -y -c anaconda scipy==1.11.3
conda install -y -c anaconda seaborn==0.12.2
conda install -y matplotlib==3.8.0
pip install wandb==0.17.1
```

## Model Download
```shell
wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
mkdir -p model/vit_checkpoint/imagenet21k
mv R50+ViT-B_16.npz model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

## Dataset Download
```shell
gdown https://drive.google.com/uc?id=1kYKUqt8UCKgr4PiSN-j_-3p48yJAgnCF
```
