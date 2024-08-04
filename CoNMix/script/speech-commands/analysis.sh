
# python -m CoNMix.speech-commands.analysis --dataset_root_path '/root/data/speech_commands' \
#     --temporary_path '/root/tmp/speech_commands/0.05-gaussian_noise' \
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --normalized --severity_level 0.05
    
python -m CoNMix.speech-commands.analysis --dataset_root_path '/root/data/speech_commands' \
    --temporary_path '/root/tmp/speech_commands/bg-10.0-doing_the_dishes' \
    --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
    --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
    --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
    --normalized --severity_level 10.0