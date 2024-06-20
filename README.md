# Test-time Adaptation in Audio Classification

## Project Structure
+ **legacy**: it includes all previous research algorithm implements

`Note`: the **legacy** is `excluded from this project's implementation`. 
You can ignore them if you do not want to review the previous research.

## Software Environment
```shell
conda create --name my-audio python=3.9 -y 
conda activate my-audio
# CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install tqdm==4.66.4 -y
conda install jupyter
conda install matplotlib==3.8.4
```
