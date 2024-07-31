
python -m CoNMix.speech-commands.pre_train --dataset_root_path '/root/data/speech_commands' --max_epoch 5 \
    --interval 20 --batch_size 64 --trte full --temporary_path '/root/tmp/speech_commands_train' \
    --normalized --wandb