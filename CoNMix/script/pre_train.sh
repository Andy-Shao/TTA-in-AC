export PYTHONPATH=$PYTHONPATH:$(pwd)

python CoNMix/pre_train.py --dataset_root_path '/root/data/AudioMNIST/data' --max_epoch 20 \
    --interval 20 --batch_size 64 --trte full --temporary_path '/root/tmp/AudioMNIST_train' \
    --wandb --test_rate 0.1