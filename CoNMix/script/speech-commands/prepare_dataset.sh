
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --output_path $BASE_PATH'/tmp/speech_commands/guassian_noise/0.005' --severity_level 0.005 --corruption 'guassian_noise' \
#     --data_type 'final' --cal_strong --dataset 'speech-commands' --parallel

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --output_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/3.0' --severity_level 3.0 --corruption 'doing_the_dishes' \
#     --data_type 'final' --cal_strong --dataset 'speech-commands' --parallel --rand_bg

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --output_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/3.0' --severity_level 3.0 --corruption 'exercise_bike' \
#     --data_type 'final' --cal_strong --dataset 'speech-commands' --parallel --rand_bg

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --output_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/3.0' --severity_level 3.0 --corruption 'running_tap' \
#     --data_type 'final' --cal_strong --dataset 'speech-commands' --parallel --rand_bg

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --output_path $BASE_PATH'/tmp/speech_commands_purity/running_tap-bg/3.0' --severity_level 3.0 --corruption 'running_tap' \
#     --data_type 'final' --cal_strong --dataset 'speech-commands-purity' --parallel --rand_bg

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --output_path $BASE_PATH'/tmp/speech_commands_purity/running_tap-bg_fixed/3.0' --severity_level 3.0 --corruption 'running_tap' \
#     --data_type 'final' --cal_strong --dataset 'speech-commands-purity' --parallel

python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --output_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/10.0' --severity_level 10.0 --corruption 'doing_the_dishes' \
    --data_type 'final' --cal_strong --dataset 'speech-commands' --parallel --rand_bg --clip

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --output_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/10.0' --severity_level 10.0 --corruption 'exercise_bike' \
#     --data_type 'final' --cal_strong --dataset 'speech-commands' --parallel --rand_bg

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --output_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/10.0' --severity_level 10.0 --corruption 'running_tap' \
#     --data_type 'final' --cal_strong --dataset 'speech-commands' --parallel --rand_bg