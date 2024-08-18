# export PYTHONPATH=$PYTHONPATH:$(pwd)
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m CoNMix.prepare_corrupted_dataset --dataset 'audio-mnist' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --output_path $BASE_PATH'/tmp/AudioMNIST_analysis_005' --severity_level 0.005 --corruption 'gaussian_noise' \
#     --cal_strong

# python -m CoNMix.prepare_corrupted_dataset --dataset 'audio-mnist' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --output_path $BASE_PATH'/tmp/AudioMNIST_analysis_0025' --severity_level 0.0025 --corruption 'gaussian_noise' \
#     --cal_norm --cal_strong --parallel

python -m CoNMix.prepare_corrupted_dataset --dataset 'audio-mnist' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --output_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/3.0-bg-rand' --severity_level 3.0 --corruption 'doing_the_dishes' \
    --background_root_path $BASE_PATH'/data/speech_commands' --cal_strong --parallel --rand_bg

# python -m CoNMix.prepare_corrupted_dataset --dataset 'audio-mnist' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --output_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/10.0-bg-rand' --severity_level 10.0 --corruption 'doing_the_dishes' \
#     --background_root_path $BASE_PATH'/data/speech_commands' --cal_strong --parallel --rand_bg

# python -m CoNMix.prepare_corrupted_dataset --dataset 'audio-mnist' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --output_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/10.0-bg-rand' --severity_level 10.0 --corruption 'exercise_bike' \
#     --background_root_path $BASE_PATH'/data/speech_commands' --cal_strong --parallel --rand_bg