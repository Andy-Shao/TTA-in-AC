# Test-time Adaptation in Audio Classification

## Project Structure
+ **legacy**: it includes all previous research algorithm implements
+ **lib**: the library code for this project.

`Note`: the **legacy** is `excluded from this project's implementation`. 
You can ignore them if you do not want to review the previous research.

## Software Environment
```shell
conda create --name my-audio python=3.9 -y 
conda activate my-audio
# CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install tqdm==4.66.4 -y
conda install jupyter -y
conda install matplotlib==3.8.4 -y
conda install pandas==2.2.2 -y 
```

## Training
### Tent & Norm Adaptation
```shell
sh pre_train.sh
```
`Note`: try to modify the `--dataset_root_path ` for your dataset location.

## Test
After that open and run the `analysis.ipynb` to demonstrate the analysis feedback. 
Most importantly, modify the `test_data_root_path` to your dataset location.

## Dataset
### Audio MNIST
+ sample size: 30000
+ sample rate: 48000
+ sample data shape: [1, 14073 - 47998]
  
[Audio MNIST Link](https://github.com/soerenab/AudioMNIST/tree/master)

## Code Reference
[Tent](https://github.com/DequanWang/tent)