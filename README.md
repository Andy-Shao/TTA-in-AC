# Test-time Adaptation in Audio Classification

## Project Structure
+ **legacy**: it includes all previous research algorithm implements
+ **lib**: the library code for this project.
+ **TTT**: the TTBA test-time training algorithm implement.
+ **CoNMix**: the CoNMix test-time training algorithm implement

`Note`: the **legacy** is `excluded from this project's implementation`. 
You can ignore them since them are abandoned implements.

## Software Environment
```shell
conda create --name my-audio python=3.9 -y 
conda activate my-audio
# CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -y -c anaconda scipy==1.11.3
conda install conda-forge::ml-collections==0.1.1 -y
conda install pandas==2.2.2 -y
conda install tqdm==4.66.4 -y
conda install jupyter -y
conda install matplotlib==3.8.4 -y 
pip install wandb==0.17.1
```

## Training
### Tent & Norm Adaptation
```shell
sh script/pre_train.sh
```
`Note`: try to modify the `--dataset_root_path ` for your dataset location.

### TTT Adaptation
```shell
sh script/ttt_pre_train.sh
```
`Note`: try to modify the `--dataset_root_path ` for your dataset location.

### CoNMix 
Model download:
```shell
wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
mkdir -p model/vit_checkpoint/imagenet21k
mv R50+ViT-B_16.npz model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

## Analysis
### Tent & Norm Adaptation
```shell
sh script/analysis.sh
```
`Note`: try to modify the `--dataset_root_path ` for your dataset location.


### TTT Adaptation
```shell
sh script/ttt_analysis.sh
```
`Note`: try to modify the `--dataset_root_path ` for your dataset location.

### Exhibition
After that open and run the `analysis_exhibition.ipynb` to demonstrate the analysis feedback. 

## Dataset
### Audio MNIST
+ sample size: 30000
+ sample rate: 48000
+ sample data shape: [1, 14073 - 47998]
  
[Audio MNIST Link](https://github.com/soerenab/AudioMNIST/tree/master)

## Code Reference
+ [tent](https://github.com/DequanWang/tent)
+ [ttt_cifar_release](https://github.com/yueatsprograms/ttt_cifar_release/tree/master)
+ [CoNMix](https://github.com/vcl-iisc/CoNMix/tree/master)
+ [TransUNet](https://github.com/Beckschen/TransUNet)
+ [SHOT](https://github.com/tim-learn/SHOT)