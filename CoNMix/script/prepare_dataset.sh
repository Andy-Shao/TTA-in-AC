# export PYTHONPATH=$PYTHONPATH:$(pwd)
export BASE_PATH='/home/andyshao'

# python -m CoNMix.prepare_corrupted_dataset --dataset 'audio-mnist' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --output_path $BASE_PATH'/tmp/AudioMNIST_analysis_005' --severity_level 0.005 --corruption 'gaussian_noise' \
#     --cal_strong

python -m CoNMix.prepare_corrupted_dataset --dataset 'audio-mnist' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --output_path $BASE_PATH'/tmp/AudioMNIST_analysis_0025' --severity_level 0.0025 --corruption 'gaussian_noise' \
    --cal_norm --cal_strong --parallel
