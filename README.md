# An Investigation of Test-time Adaptation for Audio Classification under Background Noise

## Project Structure
+ **lib**: The library code for this project.
+ **ttt**: The TTBA test-time training algorithm -- [TTT](https://yueatsprograms.github.io/ttt/home.html) is implemented.
+ **CoNMix**: The [CoNMix](https://sites.google.com/view/conmix-vcl) test-time training algorithm is implemented
+ **tent**: The OTTA test time training -- [TENT](https://doi.org/10.48550/arXiv.2006.10726) (tent adaptation, norm adaptation)

## Software Environment
- Docker image: nvidia/cuda:11.8.0-devel-ubuntu22.04
- GPU: RTX 4090
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
In some cloud platforms, such as [Google Cloud](https://cloud.google.com/). You should install more:
```shell
pip install soundfile
```

## Processing
```
export BASE_PATH={the parent path of this project}
git clone https://github.com/Andy-Shao/TTA-in-AC.git
conda activate my-audio
cd TTA-in-AC
```
[Trained weights (tar.gz file)](https://drive.google.com/file/d/1LOGKHBgUm43SC6pGq3MKq_P7weRIgIJW/view?usp=drive_link) includes the training and TTA weights of TENT, TTT, and CoNMix on AudioMNIST.
### Tent & Norm Adaptation
#### Pre-training
```shell
sh script/pre_train.sh
```
`Note`: Modify the `--dataset_root_path ` to your AudioMNIST location.
#### Analysis
```shell
sh script/analysis.sh
```
`Note`: Modify the `--dataset_root_path ` to your AudioMNIST location. Modify the `--background_root_path` to your SpeechCommands V1 location.

### TTT Adaptation
#### Pre-training
```shell
sh script/ttt_pre_train.sh
```
`Note`: Modify the `--dataset_root_path ` to your AudioMNIST location.

#### Analysis
```shell
sh script/ttt_analysis.sh
```
`Note`: Modify the `--dataset_root_path ` to your AudioMNIST location. Modify the `--background_root_path` to your SpeechCommands V1 location.

### CoNMix
Model download:
```shell
wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
mkdir -p model/vit_checkpoint/imagenet21k
mv R50+ViT-B_16.npz model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```
#### Pre-training
```shell
sh CoNMix/script/pre_train.sh
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
### AudioMNIST
+ sample size: 30000
+ sample rate: 48000
+ sample data shape: [1, 14073 - 47998]
  
[Official Audio MNIST Link](https://github.com/soerenab/AudioMNIST/tree/master)<br/>
[Hosting Download Link](https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist)
<!--[Hosting Download Link](https://drive.google.com/file/d/1kq5_qCKRUTHmViDIziSRKPjW4fIoyT9u/view?usp=drive_link)-->

### SpeechCommands V1
The dataset (1.4 GB) comprises 65,000 one-second-long utterances of 30 short words, contributed by thousands of different people through the AIY website. This is a set of one-second .wav audio files, each containing a single spoken English word.

In both versions, ten of them are used as commands by convention: "Yes", "No", "Up", "Down", "Left",
"Right", "On", "Off", "Stop", "Go". Other words are considered to be auxiliary (in the current implementation,
it is marked by the `True` value of `the "is_unknown"` feature). Their function is to teach a model to distinguish core words
from unrecognized ones.

+ Sample size: 64721 (train: 51088, test: 6835, validation: 6798)
+ sample rate: 16000
+ sample data shape: [1, 5945 - 16000]

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
+ [TENT](https://github.com/DequanWang/tent)
+ [TTT_cifar_release](https://github.com/yueatsprograms/ttt_cifar_release/tree/master)
+ [CoNMix](https://github.com/vcl-iisc/CoNMix/tree/master)
+ [TransUNet](https://github.com/Beckschen/TransUNet)
+ [SHOT](https://github.com/tim-learn/SHOT)

## Citation
```text
@article{shao2025investigation,
  title={An Investigation of Test-time Adaptation for Audio Classification under Background Noise},
  author={Shao, Weichuang and Liao, Iman Yi and Maul, Tomas Henrique Bode and Chandesa, Tissa},
  journal={arXiv preprint arXiv:2507.15523},
  year={2025}
}
```
