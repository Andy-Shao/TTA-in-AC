
python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path '/root/data/speech_commands' \
    --output_path '/root/tmp/speech_commands_20.0' --severity_level 20.0 --corruption 'doing_the_dishes' \
    --cal_strong --data_type 'raw'