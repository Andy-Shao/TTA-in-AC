export BASE_PATH='/root'

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path '/root/data/speech_commands' \
#     --temporary_path '/root/tmp/speech_commands/guassian_noise/0.05-gaussian_noise-weak' \
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --normalized --severity_level 0.05
    
# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/10.0-doing_the_dishes-weak' \
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --normalized --severity_level 10.0 --data_type 'final'

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path '/root/data/speech_commands' \
#     --temporary_path '/root/tmp/speech_commands/doing_the_dishes-bg/10.0-doing_the_dishes-weak' \
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --normalized --severity_level 3.0 --data_type 'final'

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/10.0-running_tap-weak' \
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --normalized --severity_level 10.0 --data_type 'final' --corruption 'running_tap'

python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --temporary_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/1.0-running_tap-weak' \
    --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
    --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
    --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
    --normalized --severity_level 1.0 --data_type 'final' --corruption 'running_tap' \
    --dataset 'speech-commands'

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech_commands_purity/running_tap-bg/1.0-running_tap-weak' \
#     --modelF_weight_path './result/speech-commands-purity/CoNMix/pre_train/speech-commands-purity_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands-purity/CoNMix/pre_train/speech-commands-purity_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands-purity/CoNMix/pre_train/speech-commands-purity_best_modelC.pt' \
#     --normalized --severity_level 1.0 --data_type 'final' --corruption 'running_tap' \
#     --dataset 'speech-commands-purity'