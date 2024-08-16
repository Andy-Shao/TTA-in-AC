# export PYTHONPATH=$PYTHONPATH:$(pwd)
export BASE_PATH='/home/andyshao'

# python -m ttt.time_shift_analysis --origin_model_weight_file_path './result/audio-mnist/ttt/pre_time_shift_train/ts_bn_ckpt.pth' \
#      --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --depth 26 --threshold 0.99 --shift_limit 0.2625 --severity_level 0.0025 \
#      --batch_size 126 --output_csv_name 'ts_bn_accuracy_record_0025.csv'

# python -m ttt.time_shift_analysis --origin_model_weight_file_path './result/audio-mnist/ttt/pre_time_shift_train/ts_bn_ckpt.pth' \
#      --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --depth 26 --threshold 0.99 --shift_limit 0.2625 --severity_level 0.005 \
#      --batch_size 126 --output_csv_name 'ts_bn_accuracy_record_005.csv'

# python -m ttt.bg_time_shift_analysis --origin_model_weight_file_path './result/audio-mnist/ttt/pre_time_shift_train/ts_bn_ckpt.pth' \
#      --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --depth 26 --threshold 0.99 --shift_limit 0.2625 --severity_level 1.0 \
#      --batch_size 126 --output_csv_name 'ts_bn_accuracy_record_bg-rand-1.0.csv' --background_root_path $BASE_PATH'/data/speech_commands' \
#      --corruptions 'doing_the_dishes,exercise_bike,running_tap' --rand_bg

# python -m ttt.bg_time_shift_analysis --origin_model_weight_file_path './result/audio-mnist/ttt/pre_time_shift_train/ts_bn_ckpt.pth' \
#      --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --depth 26 --threshold 0.99 --shift_limit 0.2625 --severity_level 1.0 \
#      --batch_size 126 --output_csv_name 'ts_bn_accuracy_record_bg-fixed-1.0.csv' --background_root_path $BASE_PATH'/data/speech_commands' \
#      --corruptions 'doing_the_dishes,exercise_bike,running_tap'

python -m ttt.bg_time_shift_analysis --origin_model_weight_file_path './result/audio-mnist/ttt/pre_time_shift_train/ts_bn_ckpt.pth' \
     --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --depth 26 --threshold 0.99 --shift_limit 0.2625 --severity_level 10.0 \
     --batch_size 126 --output_csv_name 'ts_bn_accuracy_record_bg-rand-10.0.csv' --background_root_path $BASE_PATH'/data/speech_commands' \
     --corruptions 'doing_the_dishes,exercise_bike,running_tap' --rand_bg

# python ttt/time_shift_analysis.py --origin_model_weight_file_path './result/audio-mnist/ttt/pre_time_shift_train/sl_gn_ckpt.pth' \
#      --dataset_root_path '/root/data/AudioMNIST/data' --depth 20 --threshold 0.99 --shift_limit 0.2625 --severity_level 0.0025 \
#      --group_norm 8 --batch_size 192 --output_csv_name 'ts_gn_accuracy_record_0025.csv'

# python ttt/angle_shift_analysis.py --origin_model_weight_file_path './result/audio-mnist/ttt/pre_angle_shift_train/as_bn_ckpt.pth' \
#      --dataset_root_path '/root/data/AudioMNIST/data' --depth 26 --threshold 0.99 --severity_level 0.0025 \
#      --output_csv_name 'as_bn_accuracy_record_0025.csv'

# python ttt/angle_shift_analysis.py --origin_model_weight_file_path './result/audio-mnist/ttt/pre_angle_shift_train/as_bn_ckpt.pth' \
#      --dataset_root_path '/root/data/AudioMNIST/data' --depth 26 --threshold 0.99 --severity_level 0.005 \
#      --output_csv_name 'as_bn_accuracy_record_005.csv'