
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m tent.speech_commands.pre_train --dataset_root_path $BASE_PATH'/data/speech_commands' --max_epoch 20 \
#     --model 'cnn' --output_weight_name 'cnn_weights.pt' --wandb

# python -m tent.speech_commands.pre_train --dataset_root_path $BASE_PATH'/data/speech_commands' --max_epoch 20 \
#     --model 'restnet50' --output_csv_name 'RestNet50_training_records.csv' \
#     --output_weight_name 'RestNet50_weight.pt' --normalized --wandb 

python -m tent.speech_commands.pre_train --dataset 'speech-commands-numbers' \
    --dataset_root_path $BASE_PATH'/data/speech_commands' --max_epoch 20 \
    --model 'restnet50' --output_csv_name 'restnet50_training_records.csv' \
    --output_weight_name 'restnet50_weight.pt' --normalized --weight_decay '5e-4' \
    --wandb 