# Test-time Adaptation in Audio Classification

## Project Structure
+ **legacy**: it includes all previous research algorithm implements
+ **lib**: the library code for this project.
+ **TTT**: the TTBA test-time training algorithm implemented.
+ **CoNMix**: the CoNMix test-time training algorithm implement

`Note`: the **legacy** is `excluded from this project's implementation`. 
You can ignore them since they are abandoned implements.

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

## Processing
### Tent & Norm Adaptation
#### Pre-train
```shell
sh script/pre_train.sh
```
`Note`: try to modify the `--dataset_root_path ` for your dataset location.
#### Analysis
```shell
sh script/analysis.sh
```
`Note`: try to modify the `--dataset_root_path ` for your dataset location.

### TTT Adaptation
#### Pre-train
```shell
sh script/ttt_pre_train.sh
```
`Note`: try to modify the `--dataset_root_path ` for your dataset location.

#### Analysis
```shell
sh script/ttt_analysis.sh
```
`Note`: try to modify the `--dataset_root_path ` for your dataset location.

### CoNMix for AudioMNIST
Model download:
```shell
wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
mkdir -p model/vit_checkpoint/imagenet21k
mv R50+ViT-B_16.npz model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```
#### Pre-train
```shell
sh CoNMix/script/pre-train.sh
```
#### Prepare The Corruption Data
```shell
sh CoNMix/script/prepare_dataset.sh
```
#### STDA
```shell
sh CoNMix/script/STDA.sh
```
#### Analysis
```shell
sh CoNMix/script/analysis.sh
```

## Exhibition
After that open and run the `analysis_exhibition.ipynb` to demonstrate the analysis feedback. 

## Dataset
### Audio MNIST
+ sample size: 30000
+ sample rate: 48000
+ sample data shape: [1, 14073 - 47998]
  
[Audio MNIST Link](https://github.com/soerenab/AudioMNIST/tree/master)

### ESC-50 Dataset for Environmental Sound Classification
+ sample size: $50 \times 40$

The ESC-50 dataset is a labelled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification.
  
The dataset consists of 5-second-long recordings organized into 50 semantical classes (with 40 examples per class) loosely arranged into 5 major categories:

| Animals | Natural soundscapes & water sounds | Human, non-speech sounds | Interior/domestic sounds | Exterior/urban noises |
|--|--|--|--|--|
|Dog|Rain|Crying baby|`Door knock`|Helicopter|
|Rooster|Sea waves|Sneezing|Mouse click|Chainsaw|
|Pig|Crackling fire|Clapping|`Keyboard typing`|Siren|
|Cow|Crickets|Breathing|Door, wood creaks|Car horn|
|Frog|Chirping birds|Coughing|Can opening|Engine|
|Cat|Water drops|Footsteps|Washing machine|`Train`|
|Hen|Wind|Laughing|Vacuum cleaner|`Church bells`|
|Insects (flying)|Pouring water|Brushing teeth|Clock alarm|Airplane|
|Sheep|Toilet flush|Snoring|Clock tick|Fireworks|
|Crow|Thunderstorm|Drinking, sipping|Glass breaking|Hand saw|

[ESC-50 Link](https://github.com/karolpiczak/ESC-50)

### Speech Commands Dataset
+ Sample size: 65,000
  
The dataset (1.4 GB) has 65,000 one-second long utterances of 30 short words by thousands of different people, contributed by public members through the AIY website. This is a set of one-second .wav audio files, each containing a single spoken English word.

[Speech Commands Dataset Link](https://research.google/blog/launching-the-speech-commands-dataset/)

## Code Reference
+ [tent](https://github.com/DequanWang/tent)
+ [ttt_cifar_release](https://github.com/yueatsprograms/ttt_cifar_release/tree/master)
+ [CoNMix](https://github.com/vcl-iisc/CoNMix/tree/master)
+ [TransUNet](https://github.com/Beckschen/TransUNet)
+ [SHOT](https://github.com/tim-learn/SHOT)
