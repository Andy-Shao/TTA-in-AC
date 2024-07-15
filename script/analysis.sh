export PYTHONPATH=$PYTHONPATH:$(pwd)
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256'

# python analysis.py --model_weight_file_path './result/audio-mnist/tent/pre_train/model_weights.pt' --dataset_root_path '/root/data/AudioMNIST/data' \
#     --severity_level 0.0025 --output_csv_name 'accuracy_record_0025.csv'

# python analysis.py --model_weight_file_path './result/audio-mnist/tent/pre_train/model_weights.pt' --dataset_root_path '/root/data/AudioMNIST/data' \
#     --severity_level 0.005 --output_csv_name 'accuracy_record_005.csv'

# python analysis.py --dataset_root_path '/root/data/AudioMNIST/data' --severity_level 0.0025 --batch_size 256 \
#     --model 'restnet50' --cal_norm

# python analysis.py --model_weight_file_path './result/audio-mnist/tent/pre_train/RestNet50_weight.pt' \
#     --dataset_root_path '/root/data/AudioMNIST/data' --severity_level 0.0025 --model 'restnet50' \
#     --test_mean '-51.242424, -51.242424, -51.242424' --test_std '19.169205, 19.169205, 19.169205' \
#     --corrupted_test_mean '-30.822489, -30.822489, -30.822489' --corrupted_test_std '7.736464, 7.736464, 7.736464' \
#     --output_csv_name 'RestNet50_accuracy_record_0025.csv' --normalized

# python analysis.py --model_weight_file_path './result/audio-mnist/tent/pre_train/RestNet50_batch_weight.pt' \
#     --dataset_root_path '/root/data/AudioMNIST/data' --severity_level 0.0025 --model 'restnet50' \
#     --output_csv_name 'RestNet50_batch_accuracy_record_0025.csv'

python analysis.py --model_weight_file_path './result/audio-mnist/tent/pre_train/RestNet50_batch_weight.pt' \
    --dataset_root_path '/root/data/AudioMNIST/data' --severity_level 0.005 --model 'restnet50' \
    --output_csv_name 'RestNet50_batch_accuracy_record_005.csv'