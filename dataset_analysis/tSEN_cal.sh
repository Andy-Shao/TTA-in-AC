export BASE_PATH=${BASE_PATH:-'\root'}

# python -m dataset_analysis.tSEN_calculation --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --dataset 'audio-mnist' --output_file 'audio-mnist_full_tsen.csv' --mode 'full'

# python -m dataset_analysis.tSEN_calculation --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --dataset 'speech-commands' --output_file 'speech-commands_full_tsen.csv' --mode 'full'

# python -m dataset_analysis.tSEN_calculation --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --dataset 'audio-mnist' --output_file 'audio-mnist_full_noreduced_tsen.csv' --mode 'full' \
#     --no_reduce

# python -m dataset_analysis.tSEN_calculation --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --dataset 'speech-commands' --output_file 'speech-commands_full_noreduced_tsen.csv' --mode 'full' \
#     --no_reduce

# python -m dataset_analysis.tSEN_calculation --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --dataset 'speech-commands-numbers' --output_file 'speech-commands-numbers_full_noreduced_tsen.csv' --mode 'full' \
#     --no_reduce

python -m dataset_analysis.tSEN_calculation --dataset_root_path $BASE_PATH'/data/cifar10' \
    --dataset 'cifar-10' --output_file 'cifar-10_full_noreduced_tsen.csv' --mode 'full' \
    --no_reduce