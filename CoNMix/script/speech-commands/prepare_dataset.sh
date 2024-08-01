
python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path '/root/data/speech_commands' \
    --output_path '/root/tmp/speech_commands_0.005' --severity_level 0.005 --corruption 'doing_the_dishes' \
    --cal_strong --data_type 'raw'