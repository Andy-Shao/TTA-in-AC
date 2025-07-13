#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m ttt.pre_time_shift_train --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --max_epoch 10 --depth 26 --shift_limit 0.2625 \
    --batch_size 126 --rotation_type 'expand' --output_csv_name 'ts_bn_accu_record.csv' --output_weight_name 'ts_bn_ckpt.pth'

# python -m ttt.pre_time_shift_train --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --max_epoch 10 --depth 20 --shift_limit 0.2625 \
#     --group_norm 8 --batch_size 192 --milestone_1 75 --milestone_2 125 \
#     --output_csv_name 'ts_gn_accu_record.csv' --output_weight_name 'ts_gn_ckpt.pth'

# python -m ttt.pre_angle_shift_train --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --max_epoch 10 --depth 26 \
#     --rotation_type 'expand' --output_csv_name 'as_bn_accu_record.csv' --output_weight_name 'as_bn_ckpt.pth'