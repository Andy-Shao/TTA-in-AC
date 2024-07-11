export PYTHONPATH=$PYTHONPATH:$(pwd)

python CoNMix/prepare_corrupted_dataset.py --dataset 'audio-mnist' --dataset_root_path '/root/data/AudioMNIST/data' \
    --output_path '/root/tmp/AudioMNIST_analysis_005' --severity_level 0.005 --corruption 'gaussian_noise' \
    --cal_norm --cal_strong
