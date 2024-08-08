export BASE_PATH='/home/andy'

python -m ttt.speech_commands.pre_time_shift_train --dataset_root_path $BASE_PATH'/data/speech_commands' --max_epoch 10 \
    --depth 50 --shift_limit 0.2625 --batch_size 32 --rotation_type 'expand' --output_csv_name 'ts_bn_accu_record.csv' \
    --output_weight_name 'ts_bn_ckpt.pth'