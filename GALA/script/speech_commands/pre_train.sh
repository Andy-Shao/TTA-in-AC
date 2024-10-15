export BASE_PATH=${BASE_PATH:-'\root'}

python -m GALA.speech_commands.pre_train --dataset 'speech-commands' \
    --dataset_root_path $BASE_PATH'/data/speech_commands' --max_epoch 20 \
    --model 'restnet50' --output_csv_name 'restnet50_training_records.csv' \
    --output_weight_name 'restnet50_weight.pt' --normalized --weight_decay '5e-4' \
    --wandb