export BASE_PATH='/root'

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path '/root/data/speech_commands' \
#     --temporary_path '/root/tmp/speech_commands/guassian_noise/0.05-gaussian_noise-weak' \
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --normalized --severity_level 0.05
    
python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --temporary_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/1.0-running_tap-weak' \
    --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
    --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
    --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
    --adapted_modelF_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelF-bg-1.0-running_tap.pt' \
    --adapted_modelB_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelB-bg-1.0-running_tap.pt' \
    --adapted_modelC_weight_path './result/speech-commands/CoNMix/STDA/speech-commands_modelC-bg-1.0-running_tap.pt' \
    --normalized --severity_level 1.0 --data_type 'final' --corruption 'running_tap' \
    --dataset 'speech-commands' --analyze_STDA --output_csv_name 'bg-1.0-running_tap_accuracy_record.csv'

# python -m CoNMix.speech-commands.STDA_analysis --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --temporary_path $BASE_PATH'/tmp/speech_commands_purity/running_tap-bg/1.0-running_tap-weak' \
#     --modelF_weight_path './result/speech-commands-purity/CoNMix/pre_train/speech-commands-purity_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands-purity/CoNMix/pre_train/speech-commands-purity_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands-purity/CoNMix/pre_train/speech-commands-purity_best_modelC.pt' \
#     --normalized --severity_level 1.0 --data_type 'final' --corruption 'running_tap' \
#     --dataset 'speech-commands-purity'