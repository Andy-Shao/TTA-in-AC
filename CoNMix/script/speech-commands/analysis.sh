
python -m CoNMix.speech-commands.analysis --dataset_root_path '/root/data/speech_commands' \
    --temporary_path '/root/tmp/speech_commands_7.0-doing_the_dishes' \
    --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
    --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
    --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
    --normalized