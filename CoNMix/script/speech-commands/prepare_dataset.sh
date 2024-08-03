python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path '/root/data/speech_commands' \
    --output_path '/root/tmp/speech_commands/0.05' --severity_level 0.05 --corruption 'gaussian_noise' \
    --data_type 'final' --cal_strong

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path '/home/andyshao/data/speech_commands' \
#     --output_path '/home/andyshao/tmp/speech_commands/bg-10.0' --severity_level 10.0 --corruption 'doing_the_dishes' \
#     --data_type 'final' --cal_strong