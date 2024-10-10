export BASE_PATH=${BASE_PATH:-'\root'}

# python -m dataset_analysis.tSEN_calculation --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --dataset 'audio-mnist' --output_file 'audio-mnist_train_tsen.csv' --mode 'train'

# python -m dataset_analysis.tSEN_calculation --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --dataset 'audio-mnist' --output_file 'audio-mnist_test_tsen.csv' --mode 'test'

# python -m dataset_analysis.tSEN_calculation --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --dataset 'speech-commands' --output_file 'speech-commands_test_tsen.csv' --mode 'test'

# python -m dataset_analysis.tSEN_calculation --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --dataset 'speech-commands' --output_file 'speech-commands_train_tsen.csv' --mode 'train'

python -m dataset_analysis.tSEN_calculation --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --dataset 'audio-mnist' --output_file 'audio-mnist_full_tsen.csv' --mode 'full'

python -m dataset_analysis.tSEN_calculation --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --dataset 'speech-commands' --output_file 'speech-commands_full_tsen.csv' --mode 'full'