# Test-time Adaptation in Audio Classification

## Project Structure
+ **lib**: the library code for this project.
+ **TTT**: the TTBA test-time training algorithm implemented.
+ **CoNMix**: the CoNMix test-time training algorithm implement
+ **tent**: the OTTA test time training (tent adaptation, norm adaptation)

## Software Environment
Machine image: nvidia/cuda:11.8.0-devel-ubuntu22.04
```shell
conda create --name my-audio python=3.9 -y 
conda activate my-audio
# CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -y -c anaconda scipy==1.11.3
conda install conda-forge::ml-collections==0.1.1 -y
conda install pandas==2.2.2 -y
# conda install conda-forge::pydub==0.25.1 -y
conda install tqdm==4.66.4 -y
conda install jupyter -y
conda install matplotlib==3.8.4 -y 
pip install wandb==0.17.1
```
In some cloud platforms, such as [Google Cloud](https://console.cloud.google.com). You should install more:
```shell
pip install soundfile
```

## Processing
```
export BASE_PATH={the parent path of this project}
conda activate my-audio
cd TTA-in-AC
```
### Tent & Norm Adaptation
#### Pre-train
```shell
sh script/pre_train.sh
```
`Note`: Modify the `--dataset_root_path ` to your AudioMNIST location.
#### Analysis
```shell
sh script/analysis.sh
```
`Note`: Modify the `--dataset_root_path ` to your AudioMNIST location. Modify the `--background_root_path` to your SpeechCommands v0.01 location.

### TTT Adaptation
#### Pre-train
```shell
sh script/ttt_pre_train.sh
```
`Note`: Modify the `--dataset_root_path ` to your AudioMNIST location.

#### Analysis
```shell
sh script/ttt_analysis.sh
```
`Note`: Modify the `--dataset_root_path ` to your AudioMNIST location. Modify the `--background_root_path` to your SpeechCommands V0.01 location.

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
`Note`: Modify the `--dataset_root_path` to your AudioMNIST location.
#### Prepare The Corruption Data
```shell
sh CoNMix/script/prepare_dataset.sh
```
`Note`: Modify the `--dataset_root_path ` to your AudioMNIST location. Modify the `--temporary_path` to your location
#### STDA
```shell
sh CoNMix/script/STDA.sh
```
#### Analysis
```shell
sh CoNMix/script/analysis.sh
```
`Note`: Modify the `--dataset_root_path ` to your AudioMNIST location. Modify the `--temporary_path` to your location

## Exhibition
After that, open and run the `analysis_exhibition.ipynb` to demonstrate the analysis feedback. 

## Dataset
### Audio MNIST
+ sample size: 30000
+ sample rate: 48000
+ sample data shape: [1, 14073 - 47998]
  
[Official Audio MNIST Link](https://github.com/soerenab/AudioMNIST/tree/master)<br/>
[Hosting Download Link](https://drive.google.com/file/d/1kq5_qCKRUTHmViDIziSRKPjW4fIoyT9u/view?usp=drive_link)

### SpeechCommands v0.01
The dataset (1.4 GB) has 65,000 one-second long utterances of 30 short words by thousands of different people, contributed by public members through the AIY website. This is a set of one-second .wav audio files, each containing a single spoken English word.

In both versions, ten of them are used as commands by convention: "Yes", "No", "Up", "Down", "Left",
"Right", "On", "Off", "Stop", "Go". Other words are considered to be auxiliary (in the current implementation
it is marked by the `True` value of `the "is_unknown"` feature). Their function is to teach a model to distinguish core words
from unrecognized ones.

+ Sample size: 64721 (train: 51088, test: 6835, validation: 6798)
+ sample rate: 16000
+ sampel data shape: [1, 5945 - 16000]

|backgroud noise type|sample data shape|sample rate|
|--|--|--|
|doing_the_dishes|[1, 1522930]|16000|
|dude_miaowing|[1, 988891]|16000|
|exercise_bike|[1, 980062]|16000|
|pink_noise|[1, 960000]|16000|
|running_tap|[1, 978488]|16000|
|white_noise|[1, 960000]|16000|

[Speech Commands Dataset Link](https://research.google/blog/launching-the-speech-commands-dataset/)<br/>
[Download Link](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)
<!-- [TensorFlow Document](https://www.tensorflow.org/datasets/community_catalog/huggingface/speech_commands) -->

## Code Reference
+ [tent](https://github.com/DequanWang/tent)
+ [ttt_cifar_release](https://github.com/yueatsprograms/ttt_cifar_release/tree/master)
+ [CoNMix](https://github.com/vcl-iisc/CoNMix/tree/master)
+ [TransUNet](https://github.com/Beckschen/TransUNet)
+ [SHOT](https://github.com/tim-learn/SHOT)
