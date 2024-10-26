export BASE_PATH=${BASE_PATH:-'/root'}

# python -m ttt.speech_commands.time_shift_analysis \
#     --origin_model_weight_file_path './result/speech-commands/ttt/pre_time_shift_train/ts_bn_ckpt.pth' \
#     --dataset 'speech-commands' --output_csv_name 'gaussian_noise_0.005-accuracy_record.csv' \
#     --dataset_root_path $BASE_PATH'/data/speech_commands' --depth 26 --severity_level 0.005 --batch_size 96 \
#     --corruptions 'gaussian_noise'

# python -m ttt.speech_commands.time_shift_analysis \
#     --origin_model_weight_file_path './result/speech-commands/ttt/pre_time_shift_train/ts_bn_ckpt.pth' \
#     --dataset 'speech-commands' --output_csv_name 'background_3.0-accuracy_record.csv' \
#     --dataset_root_path $BASE_PATH'/data/speech_commands' --depth 26 --severity_level 3.0 --batch_size 126 \
#     --corruptions 'doing_the_dishes,exercise_bike,running_tap' \
#     --TTT_analysis --lr 1e-3 --threshold 0.99 --shift_limit 0.2625

python -m ttt.speech_commands.time_shift_analysis \
    --origin_model_weight_file_path './result/speech-commands/ttt/pre_time_shift_train/ts_bn_ckpt.pth' \
    --dataset 'speech-commands' --output_csv_name 'background_3.0-accuracy_record-offline.csv' \
    --dataset_root_path $BASE_PATH'/data/speech_commands' --depth 26 --severity_level 3.0 --batch_size 126 \
    --corruptions 'doing_the_dishes,exercise_bike,running_tap' \
    --TTT_analysis --lr 1e-3 --threshold 0.99 --shift_limit 0.2625 --mode 'offline' --niter 10

# python -m ttt.speech_commands.time_shift_analysis \
#     --origin_model_weight_file_path './result/speech-commands/ttt/pre_time_shift_train/ts_bn_ckpt.pth' \
#     --dataset 'speech-commands' --output_csv_name 'background_10.0-accuracy_record.csv' \
#     --dataset_root_path $BASE_PATH'/data/speech_commands' --depth 26 --severity_level 10.0 --batch_size 126 \
#     --corruptions 'doing_the_dishes,exercise_bike,running_tap' \
#     --TTT_analysis --lr 1e-3 --threshold 0.99 --shift_limit 0.2625

python -m ttt.speech_commands.time_shift_analysis \
    --origin_model_weight_file_path './result/speech-commands/ttt/pre_time_shift_train/ts_bn_ckpt.pth' \
    --dataset 'speech-commands' --output_csv_name 'background_10.0-accuracy_record-offline.csv' \
    --dataset_root_path $BASE_PATH'/data/speech_commands' --depth 26 --severity_level 10.0 --batch_size 126 \
    --corruptions 'doing_the_dishes,exercise_bike,running_tap' \
    --TTT_analysis --lr 1e-3 --threshold 0.99 --shift_limit 0.2625 --mode 'offline' --niter 10

# python -m ttt.speech_commands.time_shift_analysis \
#     --origin_model_weight_file_path './result/speech-commands-random/ttt/pre_time_shift_train/ts_bn_ckpt.pth' \
#     --dataset 'speech-commands-random' --output_csv_name 'gaussian_noise_0.005-accuracy_record.csv' \
#     --dataset_root_path $BASE_PATH'/data/speech_commands' --depth 50 --severity_level 0.005 --batch_size 96 \
#     --corruptions 'gaussian_noise' --normalized

# python -m ttt.speech_commands.time_shift_analysis \
#     --origin_model_weight_file_path './result/speech-commands-random/ttt/pre_time_shift_train/ts_bn_ckpt.pth' \
#     --dataset 'speech-commands-random' --output_csv_name 'background_10.0-accuracy_record.csv' \
#     --dataset_root_path $BASE_PATH'/data/speech_commands' --depth 50 --severity_level 10.0 --batch_size 126 \
#     --corruptions 'doing_the_dishes,exercise_bike,running_tap' \
#     --TTT_analysis --lr 1e-3 --threshold 0.99 --shift_limit 0.2625

# python -m ttt.speech_commands.time_shift_analysis \
#     --origin_model_weight_file_path './result/speech-commands-random/ttt/pre_time_shift_train/ts_bn_ckpt.pth' \
#     --dataset 'speech-commands-random' --output_csv_name 'background_3.0-accuracy_record.csv' \
#     --dataset_root_path $BASE_PATH'/data/speech_commands' --depth 50 --severity_level 3.0 --batch_size 126 \
#     --corruptions 'doing_the_dishes,exercise_bike,running_tap' \
#     --TTT_analysis --lr 1e-3 --threshold 0.99 --shift_limit 0.2625

# python -m ttt.speech_commands.time_shift_analysis \
#     --origin_model_weight_file_path './result/speech-commands-numbers/ttt/pre_time_shift_train/ts_bn_ckpt.pth' \
#     --dataset 'speech-commands-numbers' --output_csv_name 'background_10.0-accuracy_record.csv' \
#     --dataset_root_path $BASE_PATH'/data/speech_commands' --depth 50 --severity_level 10.0 --batch_size 126 \
#     --corruptions 'doing_the_dishes,exercise_bike,running_tap' \
#     --TTT_analysis --lr 1e-3 --threshold 0.99 --shift_limit 0.2625