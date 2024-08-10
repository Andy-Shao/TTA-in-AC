export BASE_PATH='/root'

python -m CoNMix.speech-commands.pre_train --dataset_root_path $BASE_PATH'/data/speech_commands' --max_epoch 20 \
    --interval 20 --batch_size 64 --trte full --temporary_path $BASE_PATH'/tmp/speech_commands_train' \
    --normalized --dataset 'speech-commands' --output_weight_prefix 'speech-commands' \
    --num_workers 16 --wandb

# python -m CoNMix.speech-commands.pre_train --dataset_root_path $BASE_PATH'/data/speech_commands' --max_epoch 10 \
#     --interval 20 --batch_size 64 --trte full --temporary_path $BASE_PATH'/tmp/speech_commands_purity_train' \
#     --normalized --dataset 'speech-commands-purity' --output_weight_prefix 'speech-commands-purity' --wandb