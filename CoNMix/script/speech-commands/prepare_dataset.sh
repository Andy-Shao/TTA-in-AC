
export BASE_PATH='/root'

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path '/root/data/speech_commands' \
#     --output_path '/root/tmp/speech_commands/guassian_noise/0.05' --severity_level 0.05 --corruption 'gaussian_noise' \
#     --data_type 'final' --cal_strong

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --output_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/10.0' --severity_level 10.0 --corruption 'doing_the_dishes' \
#     --data_type 'final' --cal_strong

python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --output_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/1.0' --severity_level 1.0 --corruption 'exercise_bike' \
    --data_type 'final' --cal_strong --dataset 'speech-commands' --parallel

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --output_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/1.0' --severity_level 1.0 --corruption 'running_tap' \
#     --data_type 'final' --cal_strong --dataset 'speech-commands'

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --output_path $BASE_PATH'/tmp/speech_commands_purity/running_tap-bg/1.0' --severity_level 1.0 --corruption 'running_tap' \
#     --data_type 'final' --cal_strong --dataset 'speech-commands-purity'