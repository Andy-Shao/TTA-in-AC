export PYTHONPATH=$PYTHONPATH:$(pwd)

# python analysis.py --model_weight_file_path './result/audio-mnist/cnn/pre_train/model_weights.pt' --dataset_root_path '/root/data/AudioMNIST/data' \
#     --severity_level 0.0025 --output_csv_name 'accuracy_record_0025.csv'

python analysis.py --model_weight_file_path './result/audio-mnist/cnn/pre_train/model_weights.pt' --dataset_root_path '/root/data/AudioMNIST/data' \
    --severity_level 0.005 --output_csv_name 'accuracy_record_005.csv'
