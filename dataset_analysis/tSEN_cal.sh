export BASE_PATH=${BASE_PATH:-'\root'}

python -m dataset_analysis.tSEN_calculation --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --dataset 'audio-mnist' --output_file 'audio-mnist-tsen.csv'