# export PYTHONPATH=$PYTHONPATH:$(pwd)
export BASE_PATH='/root'

# python -m CoNMix.prepare_corrupted_dataset --dataset 'audio-mnist' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --output_path $BASE_PATH'/tmp/AudioMNIST_analysis_005' --severity_level 0.005 --corruption 'gaussian_noise' \
#     --cal_strong

# python -m CoNMix.prepare_corrupted_dataset --dataset 'audio-mnist' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --output_path $BASE_PATH'/tmp/AudioMNIST_analysis_0025' --severity_level 0.0025 --corruption 'gaussian_noise' \
#     --cal_norm --cal_strong --parallel

python -m CoNMix.prepare_corrupted_dataset --dataset 'audio-mnist' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --output_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/1.0-bg-rand' --severity_level 1.0 --corruption 'doing_the_dishes' \
    --background_root_path $BASE_PATH'/data/speech_commands' --cal_norm --cal_strong --parallel --rand_bg