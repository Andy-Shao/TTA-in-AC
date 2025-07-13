#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m CoNMix.pre_train --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --max_epoch 20 \
    --interval 20 --batch_size 64 --trte full --temporary_path $BASE_PATH'/tmp/AudioMNIST_train' \
    --wandb --test_rate 0.1