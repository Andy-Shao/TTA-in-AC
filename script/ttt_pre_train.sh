export PYTHONPATH=$PYTHONPATH:$(pwd)

python ttt/pre_time_shift_train.py --dataset_root_path '/root/data/AudioMNIST/data' --max_epoch 10 --depth 20 --shift_limit 0.2625 \
    --rotation_type 'expand' --output_csv_name 'ts_bn_accu_record.csv' --output_weight_name 'ts_bn_ckpt.pth'

# python ttt/pre_time_shift_train.py --dataset_root_path '/root/data/AudioMNIST/data' --max_epoch 10 --depth 20 --shift_limit 0.2625 \
#     --group_norm 8 --batch_size 192 --milestone_1 75 --milestone_2 125 \
#     --output_csv_name 'ts_gn_accu_record.csv' --output_weight_name 'ts_gn_ckpt.pth'