export BASE_PATH=${BASE_PATH:-'\root'}

# python -m dataset_analysis.tSEN_calculation --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --dataset 'audio-mnist' --output_file 'audio-mnist-tsen.csv'

python -m dataset_analysis.tSEN_calculation --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --dataset 'speech-commands' --output_file 'speech-commands-tsen.csv'