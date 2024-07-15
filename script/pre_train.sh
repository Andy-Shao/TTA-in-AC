export PYTHONPATH=$PYTHONPATH:$(pwd)

# python pre_train.py --dataset_root_path '/root/data/AudioMNIST/data' --output_path './result' --max_epoch 5 

# python pre_train.py --dataset_root_path '/root/data/AudioMNIST/data' --output_path './result' --batch_size 256\
#     --model 'restnet50' --cal_norm

python pre_train.py --dataset_root_path '/root/data/AudioMNIST/data' --output_path './result' --max_epoch 5 \
    --model 'restnet50' --train_mean '-53.39211, -53.39211, -53.39211' --train_std '19.50148, 19.50148, 19.50148' \
    --val_mean '-52.2802, -52.2802, -52.2802' --val_std '19.176311, 19.176311, 19.176311' \
    --output_csv_name 'RestNet50_training_records.csv' --output_weight_name 'RestNet50_weight.pt'