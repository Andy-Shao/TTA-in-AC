export BASE_PATH=${BASE_PATH:-'/root'}

# python -m ttt.speech_commands.pre_time_shift_train --dataset_root_path $BASE_PATH'/data/speech_commands' --max_epoch 20 \
#     --depth 50 --shift_limit 0.2625 --batch_size 96 --rotation_type 'expand' --output_csv_name 'ts_bn_accu_record.csv' \
#     --output_weight_name 'ts_bn_ckpt.pth' --milestone_1 5 --milestone_2 10 --wandb

# python -m ttt.speech_commands.pre_time_shift_train --dataset_root_path $BASE_PATH'/data/speech_commands' --max_epoch 20 \
#     --depth 50 --shift_limit 0.2625 --batch_size 96 --rotation_type 'expand' --output_csv_name 'l3_ts_bn_accu_record.csv' \
#     --output_weight_name 'l3_ts_bn_ckpt.pth' --milestone_1 5 --milestone_2 10 --shared 'layer3' --wandb

python -m ttt.speech_commands.pre_time_shift_train --dataset 'speech-commands-numbers' \
    --dataset_root_path $BASE_PATH'/data/speech_commands' --max_epoch 20 \
    --depth 50 --shift_limit 0.2625 --batch_size 96 --rotation_type 'expand' --output_csv_name 'ts_bn_accu_record.csv' \
    --output_weight_name 'ts_bn_ckpt.pth' --milestone_1 5 --milestone_2 10 --wandb