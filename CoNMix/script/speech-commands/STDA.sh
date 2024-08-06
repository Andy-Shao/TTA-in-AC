export BASE_PATH='/root'

# python -m CoNMix.speech-commands.STDA --weak_aug_dataset_root_path '/root/tmp/speech_commands/0.05-gaussian_noise-weak' \
#     --strong_aug_dataset_root_path '/root/tmp/speech_commands/0.05-gaussian_noise-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 0.05 --max_epoch 50 --interval 50 --lr '2.5e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-0.005-gaussian_noise.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-0.005-gaussian_noise.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-0.005-gaussian_noise.pt' --normalized \
#     --data_type 'final' --wandb --const_par 0.2 --fbnm_par 4.0 --cls_par 0.0 --corruption 'gaussian_noise' \
#     --alpha 0.9 --initc_num 1


# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path '/root/tmp/speech_commands/doing_the_dishes-bg/10.0-doing_the_dishes-weak' \
#     --strong_aug_dataset_root_path '/root/tmp/speech_commands/doing_the_dishes-bg/10.0-doing_the_dishes-strong' \
#     --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-doing_the_dishes.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-doing_the_dishes.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-doing_the_dishes.pt' --normalized \
#     --data_type 'final' --wandb --const_par 0.2 --fbnm_par 4.0 --cls_par 0.0 --corruption 'doing_the_dishes' \
#     --alpha 0.9 --initc_num 1

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path '/root/tmp/speech_commands/exercise_bike-bg/10.0-exercise_bike-weak' \
#     --strong_aug_dataset_root_path '/root/tmp/speech_commands/exercise_bike-bg/10.0-exercise_bike-strong' \
#     --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-exercise_bike.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-exercise_bike.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-exercise_bike.pt' --normalized \
#     --data_type 'final' --wandb --const_par 0.2 --fbnm_par 4.0 --cls_par 0.0 --corruption 'exercise_bike' \
#     --alpha 0.9 --initc_num 1

python -m CoNMix.speech-commands.STDA \
    --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/10.0-running_tap-weak' \
    --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/10.0-running_tap-strong' \
    --batch_size 32 --test_batch_size 96 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '5e-5'\
    --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
    --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
    --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
    --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-running_tap.pt' \
    --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-running_tap.pt' \
    --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-running_tap.pt' --normalized \
    --data_type 'final' --wandb --const_par 0.2 --fbnm_par 4.0 --cls_par 0.0 --corruption 'running_tap' \
    --alpha 0.9 --initc_num 1 