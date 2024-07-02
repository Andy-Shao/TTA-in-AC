export PYTHONPATH=$PYTHONPATH:$(pwd)

python analysis.py --model_weight_file_path './result/audio-mnist/cnn/pre_train/model_weights.pt' --dataset_root_path '/root/data/AudioMNIST/data'

# python ttt/time_shift_analysis.py --origin_model_weight_file_path './result/audio-mnist/ttt/pre_time_shift_train/ckpt.pth' \
#      --dataset_root_path '/root/data/AudioMNIST/data' --depth 20 --threshold 0.99 --shift_limit 0.2625 --severity_level 0.0025 \
#      --output_csv_name 'accuracy_record_0025.csv'

# python ttt/time_shift_analysis.py --origin_model_weight_file_path './result/audio-mnist/ttt/pre_time_shift_train/ckpt.pth' \
#      --dataset_root_path '/root/data/AudioMNIST/data' --depth 20 --threshold 0.99 --shift_limit 0.2625 --severity_level 0.005 \
#      --output_csv_name 'accuracy_record_005.csv'