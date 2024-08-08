export BASE_PATH='/home/andyshao'

python -m ttt.speech_commands.pre_time_shift_train --dataset_root_path $BASE_PATH'/data/speech_commands' --max_epoch 20 \
    --depth 50 --shift_limit 0.2625 --batch_size 96 --rotation_type 'expand' --output_csv_name 'ts_bn_accu_record.csv' \
    --output_weight_name 'ts_bn_ckpt.pth' --wandb