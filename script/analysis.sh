# export PYTHONPATH=$PYTHONPATH:$(pwd)
# export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256'
export BASE_PATH=${BASE_PATH:-'/home/andyshao'}

# python -m tent.analysis --model_weight_file_path './result/audio-mnist/tent/pre_train/model_weights.pt' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --severity_level 0.0025 --output_csv_name 'accuracy_record_0025.csv'

# python -m tent.analysis --model_weight_file_path './result/audio-mnist/tent/pre_train/model_weights.pt' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --severity_level 0.005 --output_csv_name 'accuracy_record_005.csv'

# python -m tent.analysis --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --severity_level 0.0025 --batch_size 256 \
#     --model 'restnet50' --cal_norm

# python -m tent.analysis --model_weight_file_path './result/audio-mnist/tent/pre_train/RestNet50_weight.pt' \
#     --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --severity_level 0.0025 --model 'restnet50' \
#     --test_mean '-51.242424, -51.242424, -51.242424' --test_std '19.169205, 19.169205, 19.169205' \
#     --corrupted_test_mean '-30.822489, -30.822489, -30.822489' --corrupted_test_std '7.736464, 7.736464, 7.736464' \
#     --output_csv_name 'RestNet50_accuracy_record_0025.csv' --normalized

# python -m tent.analysis --model_weight_file_path './result/audio-mnist/tent/pre_train/RestNet50_batch_weight.pt' \
#     --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --severity_level 0.0025 --model 'restnet50' \
#     --output_csv_name 'RestNet50_batch_accuracy_record_0025.csv'

# python -m tent.analysis --model_weight_file_path './result/audio-mnist/tent/pre_train/RestNet50_batch_weight.pt' \
#     --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --severity_level 0.005 --model 'restnet50' \
#     --output_csv_name 'RestNet50_batch_accuracy_record_005.csv'

# python -m tent.bg_analysis --model_weight_file_path './result/audio-mnist/tent/pre_train/RestNet50_batch_weight.pt' \
#     --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --severity_level 3.0 --model 'restnet50' \
#     --output_csv_name 'RestNet50_batch_accuracy_record_bg-rand-3.0.csv' --corruptions 'doing_the_dishes,exercise_bike,running_tap' \
#     --background_root_path $BASE_PATH'/data/speech_commands' --rand_bg

python -m tent.bg_analysis --model_weight_file_path './result/audio-mnist/tent/pre_train/model_weights.pt' \
    --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --severity_level 3.0 --model 'cnn' \
    --output_csv_name 'cnn_accuracy_record_bg-rand-3.0.csv' --corruptions 'doing_the_dishes,exercise_bike,running_tap' \
    --background_root_path $BASE_PATH'/data/speech_commands' --rand_bg

# python -m tent.bg_analysis --model_weight_file_path './result/audio-mnist/tent/pre_train/RestNet50_batch_weight.pt' \
#     --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --severity_level 10.0 --model 'restnet50' \
#     --output_csv_name 'RestNet50_batch_accuracy_record_bg-rand-10.0.csv' --corruptions 'doing_the_dishes,exercise_bike,running_tap' \
#     --background_root_path $BASE_PATH'/data/speech_commands' --rand_bg

# python -m tent.bg_analysis --model_weight_file_path './result/audio-mnist/tent/pre_train/model_weights.pt' \
#     --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --severity_level 10.0 --model 'cnn' \
#     --output_csv_name 'cnn_accuracy_record_bg-rand-10.0.csv' --corruptions 'doing_the_dishes,exercise_bike,running_tap' \
#     --background_root_path $BASE_PATH'/data/speech_commands' --rand_bg
