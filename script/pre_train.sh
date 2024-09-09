# export PYTHONPATH=$PYTHONPATH:$(pwd)
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m tent.pre_train --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --output_path './result' --max_epoch 5 

# python -m tent.pre_train --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --output_path './result' --batch_size 256\
#     --model 'restnet50' --cal_norm

# python -m tent.pre_train --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --output_path './result' --max_epoch 10 \
#     --model 'restnet50' --train_mean '-53.407852, -53.407852, -53.407852' --train_std '19.496902, 19.496902, 19.496902' \
#     --val_mean '-52.204514, -52.204514, -52.204514' --val_std '19.183237, 19.183237, 19.183237' \
#     --output_csv_name 'RestNet50_training_records.csv' --output_weight_name 'RestNet50_weight.pt' --normalized

python -m tent.pre_train --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --output_path './result' --max_epoch 10 \
    --model 'restnet50' --output_csv_name 'RestNet50_batch_training_records.csv' --output_weight_name 'RestNet50_batch_weight.pt'