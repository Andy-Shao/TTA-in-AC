
# python -m tent.speech_commands.pre_train --dataset_root_path '/root/data/speech_commands' --max_epoch 20 \
#     --model 'cnn' --output_weight_name 'cnn_weights.pt' --wandb

# python -m tent.speech_commands.pre_train --dataset_root_path '/root/data/speech_commands' \
#     --model 'restnet50' --cal_norm --batch_size 256

python -m tent.speech_commands.pre_train --dataset_root_path '/root/data/speech_commands' --max_epoch 20 \
    --model 'restnet50' --train_mean '-24.764997, -24.764997, -24.764997' --train_std '18.940367, 18.940367, 18.940367' \
    --val_mean '-27.528795, -27.528795, -27.528795' --val_std '18.578714, 18.578714, 18.578714' \
    --output_csv_name 'RestNet50_training_records.csv' --output_weight_name 'RestNet50_weight.pt' --normalized \
    --wandb