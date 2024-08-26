export BASE_PATH=${BASE_PATH:-'/root'}
export audio_mnist_root_path=$BASE_PATH'/dataset/AudioMNIST/data/'
export speech_commands_root_path=$BASE_PATH'/dataset/speech_commands'

python -m dataset_analysis.dataset_analysis --datasets 'audio-mnist,speech_commands,speech_commands_numbers' \
    --dataset_root_pathes $audio_mnist_root_path','$speech_commands_root_path','$speech_commands_root_path