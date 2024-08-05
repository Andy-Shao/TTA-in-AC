# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path '/root/data/speech_commands' \
#     --output_path '/root/tmp/speech_commands/guassian_noise/0.05' --severity_level 0.05 --corruption 'gaussian_noise' \
#     --data_type 'final' --cal_strong

# python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path '/root/data/speech_commands' \
#     --output_path '/root/tmp/speech_commands/doing_the_dishes-bg/10.0' --severity_level 10.0 --corruption 'doing_the_dishes' \
#     --data_type 'final' --cal_strong

python -m CoNMix.speech-commands.prepare_corrupted_dataset --dataset_root_path '/root/data/speech_commands' \
    --output_path '/root/tmp/speech_commands/doing_the_dishes-bg/3.0' --severity_level 3.0 --corruption 'doing_the_dishes' \
    --data_type 'final' --cal_strong