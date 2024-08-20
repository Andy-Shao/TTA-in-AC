export BASE_PATH=${BASE_PATH:-'/root'}

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/3.0-doing_the_dishes-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/3.0-doing_the_dishes-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-3.0-doing_the_dishes.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-3.0-doing_the_dishes.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-3.0-doing_the_dishes.pt' --normalized \
#     --data_type 'final' --const_par 0.2 --fbnm_par 6.0 --cls_par 0.2 --corruption 'doing_the_dishes' --plr 1 \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_nll' --lr_gamma 30 --wandb

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/3.0-doing_the_dishes-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/3.0-doing_the_dishes-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-3.0-doing_the_dishes.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-3.0-doing_the_dishes.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-3.0-doing_the_dishes.pt' --normalized \
#     --data_type 'final' --const_par 0.2 --fbnm_par 6.0 --cls_par 0.2 --corruption 'doing_the_dishes' \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_nll' --lr_gamma 30 --wandb

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/3.0-exercise_bike-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/3.0-exercise_bike-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-3.0-exercise_bike.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-3.0-exercise_bike.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-3.0-exercise_bike.pt' --normalized \
#     --data_type 'final' --const_par 0.2 --fbnm_par 6.0 --cls_par 0.2 --corruption 'exercise_bike' \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_nll' --lr_gamma 30 --wandb

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/3.0-running_tap-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/3.0-running_tap-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-3.0-running_tap.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-3.0-running_tap.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-3.0-running_tap.pt' --normalized \
#     --data_type 'final' --wandb --const_par 0.2 --fbnm_par 6.0 --cls_par 0.2 --corruption 'running_tap' \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_nll' --lr_gamma 30

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands_purity/running_tap-bg/3.0-running_tap-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands_purity/running_tap-bg/3.0-running_tap-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 3.0 --max_epoch 100 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands-purity/CoNMix/pre_train/speech-commands-purity_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands-purity/CoNMix/pre_train/speech-commands-purity_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands-purity/CoNMix/pre_train/speech-commands-purity_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands-purity_modelF-bg-3.0-running_tap.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands-purity_modelB-bg-3.0-running_tap.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands-purity_modelC-bg-3.0-running_tap.pt' --normalized \
#     --data_type 'final' --const_par 0.2 --fbnm_par 6.0 --cls_par 0.2 --corruption 'running_tap' \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands-purity' --cls_mode 'logsoft_nll' --wandb

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/10.0-doing_the_dishes-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/10.0-doing_the_dishes-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-doing_the_dishes.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-doing_the_dishes.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-doing_the_dishes.pt' --normalized \
#     --data_type 'final' --const_par 0.2 --fbnm_par 6.0 --cls_par 0.2 --corruption 'doing_the_dishes' --plr 1 \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_nll' --lr_gamma 30 --wandb

python -m CoNMix.speech-commands.STDA \
    --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/10.0-exercise_bike-weak' \
    --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/10.0-exercise_bike-strong' \
    --batch_size 32 --test_batch_size 96 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
    --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
    --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
    --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
    --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-exercise_bike.pt' \
    --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-exercise_bike.pt' \
    --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-exercise_bike.pt' --normalized \
    --data_type 'final' --const_par 0.2 --fbnm_par 6.0 --cls_par 0.2 --corruption 'exercise_bike' \
    --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_nll' --lr_gamma 30 --wandb

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/10.0-running_tap-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/10.0-running_tap-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-running_tap.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-running_tap.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-running_tap.pt' --normalized \
#     --data_type 'final' --wandb --const_par 0.2 --fbnm_par 6.0 --cls_par 0.2 --corruption 'running_tap' \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_nll' --lr_gamma 30