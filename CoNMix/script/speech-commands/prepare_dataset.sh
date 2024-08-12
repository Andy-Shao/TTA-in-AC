
export BASE_PATH='/home/andyshao'

python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --output_path $BASE_PATH'/tmp/speech_commands/guassian_noise/0.005' --severity_level 0.005 --corruption 'guassian_noise' \
    --data_type 'final' --cal_strong --dataset 'speech-commands' --parallel

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --output_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/1.0' --severity_level 1.0 --corruption 'doing_the_dishes' \
#     --data_type 'final' --cal_strong --dataset 'speech-commands' --parallel

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --output_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/1.0' --severity_level 1.0 --corruption 'exercise_bike' \
#     --data_type 'final' --cal_strong --dataset 'speech-commands' --parallel

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --output_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/1.0' --severity_level 1.0 --corruption 'running_tap' \
#     --data_type 'final' --cal_strong --dataset 'speech-commands'

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --output_path $BASE_PATH'/tmp/speech_commands_purity/running_tap-bg/1.0' --severity_level 1.0 --corruption 'running_tap' \
#     --data_type 'final' --cal_strong --dataset 'speech-commands-purity'