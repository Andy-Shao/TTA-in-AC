export BASE_PATH=${BASE_PATH:-'/root'}

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/3.0-doing_the_dishes-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/3.0-doing_the_dishes-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-3.0-DD-org.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-3.0-DD-org.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-3.0-DD-org.pt' --normalized \
#     --data_type 'final' --const_par 0.2 --fbnm_par 4.0 --cls_par 0.2 --corruption 'doing_the_dishes' \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 --wandb \
#     --wandb_name 'DD-3dB-org' --backup_weight 1

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/3.0-doing_the_dishes-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/3.0-doing_the_dishes-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-3.0-DD-abl.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-3.0-DD-abl.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-3.0-DD-abl.pt' --normalized \
#     --data_type 'final' --const_par 0.0 --fbnm_par 4.0 --cls_par 0.2 --corruption 'doing_the_dishes' \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 --wandb \
#     --wandb_name 'DD-3dB-abl-no_cst' --early_stop 14 --backup_weight 1

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/3.0-doing_the_dishes-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/3.0-doing_the_dishes-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-3.0-DD-abl.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-3.0-DD-abl.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-3.0-DD-abl.pt' --normalized \
#     --data_type 'final' --const_par 0.2 --fbnm_par 0.0 --cls_par 0.2 --corruption 'doing_the_dishes' \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 --wandb \
#     --wandb_name 'DD-3dB-abl-no_nm' --early_stop 14 --backup_weight 1

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/3.0-doing_the_dishes-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/3.0-doing_the_dishes-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-3.0-DD-abl.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-3.0-DD-abl.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-3.0-DD-abl.pt' --normalized \
#     --data_type 'final' --const_par 0.2 --fbnm_par 4.0 --cls_par 0.0 --corruption 'doing_the_dishes' \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 --wandb \
#     --wandb_name 'DD-3dB-abl-no_pl' --early_stop 14 --backup_weight 1

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/3.0-exercise_bike-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/3.0-exercise_bike-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-3.0-EB-org.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-3.0-EB-org.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-3.0-EB-org.pt' --normalized \
#     --data_type 'final' --const_par 0.2 --fbnm_par 4.0 --cls_par 0.2 --corruption 'exercise_bike' \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 --wandb \
#     --wandb_name 'EB-3dB-org' ----early_stop 14 --backup_weight 1

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/3.0-running_tap-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/3.0-running_tap-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-3.0-RT-org.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-3.0-RT-org.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-3.0-RT-org.pt' --normalized \
#     --data_type 'final' --wandb --const_par 0.2 --fbnm_par 4.0 --cls_par 0.2 --corruption 'running_tap' \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 \
#     --wandb_name 'RT-3dB-org' --backup_weight 1

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/10.0-doing_the_dishes-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/10.0-doing_the_dishes-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-DD.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-DD.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-DD.pt' --normalized \
#     --data_type 'final' --const_par 0.2 --fbnm_par 4.0 --cls_par 0.2 --corruption 'doing_the_dishes' --plr 1 \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 --wandb \
#     --wandb_name 'DD-10dB-org' --backup_weight 1 --early_stop 20

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/10.0-doing_the_dishes-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/10.0-doing_the_dishes-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-DD.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-DD.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-DD.pt' --normalized \
#     --data_type 'final' --const_par 0.2 --fbnm_par 4.0 --cls_par 0.0 --corruption 'doing_the_dishes' --plr 1 \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 --wandb \
#     --wandb_name 'DD-10dB-no_pl' --backup_weight 1 --early_stop 20

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/10.0-doing_the_dishes-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/10.0-doing_the_dishes-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-DD.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-DD.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-DD.pt' --normalized \
#     --data_type 'final' --const_par 0.0 --fbnm_par 4.0 --cls_par 0.2 --corruption 'doing_the_dishes' --plr 1 \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 --wandb \
#     --wandb_name 'DD-10dB-no_cst' --backup_weight 1 --early_stop 20

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/10.0-doing_the_dishes-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/doing_the_dishes-bg/10.0-doing_the_dishes-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-DD.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-DD.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-DD.pt' --normalized \
#     --data_type 'final' --const_par 0.2 --fbnm_par 0.0 --cls_par 0.2 --corruption 'doing_the_dishes' --plr 1 \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 --wandb \
#     --wandb_name 'DD-10dB-no_nm' --backup_weight 1 --early_stop 20

python -m CoNMix.speech-commands.STDA \
    --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/10.0-exercise_bike-weak' \
    --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/10.0-exercise_bike-strong' \
    --batch_size 32 --test_batch_size 96 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
    --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
    --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
    --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
    --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-EB.pt' \
    --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-EB.pt' \
    --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-EB.pt' --normalized \
    --data_type 'final' --const_par 0.2 --fbnm_par 4.0 --cls_par 0.2 --corruption 'exercise_bike' \
    --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 --wandb \
    --wandb_name 'EB-10dB-org' --backup_weight 1 --early_stop 20

python -m CoNMix.speech-commands.STDA \
    --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/10.0-exercise_bike-weak' \
    --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/10.0-exercise_bike-strong' \
    --batch_size 32 --test_batch_size 96 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
    --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
    --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
    --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
    --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-EB.pt' \
    --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-EB.pt' \
    --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-EB.pt' --normalized \
    --data_type 'final' --const_par 0.2 --fbnm_par 4.0 --cls_par 0.0 --corruption 'exercise_bike' \
    --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 --wandb \
    --wandb_name 'EB-10dB-no_pl' --backup_weight 1 --early_stop 20

python -m CoNMix.speech-commands.STDA \
    --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/10.0-exercise_bike-weak' \
    --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/10.0-exercise_bike-strong' \
    --batch_size 32 --test_batch_size 96 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
    --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
    --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
    --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
    --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-EB.pt' \
    --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-EB.pt' \
    --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-EB.pt' --normalized \
    --data_type 'final' --const_par 0.0 --fbnm_par 4.0 --cls_par 0.2 --corruption 'exercise_bike' \
    --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 --wandb \
    --wandb_name 'EB-10dB-no_cst' --backup_weight 1 --early_stop 20

python -m CoNMix.speech-commands.STDA \
    --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/10.0-exercise_bike-weak' \
    --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/exercise_bike-bg/10.0-exercise_bike-strong' \
    --batch_size 32 --test_batch_size 96 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
    --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
    --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
    --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
    --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-EB.pt' \
    --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-EB.pt' \
    --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-EB.pt' --normalized \
    --data_type 'final' --const_par 0.2 --fbnm_par 0.0 --cls_par 0.2 --corruption 'exercise_bike' \
    --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 --wandb \
    --wandb_name 'EB-10dB-no_nm' --backup_weight 1 --early_stop 20

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/10.0-running_tap-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/10.0-running_tap-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-RT.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-RT.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-RT.pt' --normalized \
#     --data_type 'final' --wandb --const_par 0.2 --fbnm_par 4.0 --cls_par 0.0 --corruption 'running_tap' \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 \
#     --wandb_name 'RT-10dB-no_pl' --backup_weight 1 --early_stop 20

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/10.0-running_tap-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/10.0-running_tap-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-RT.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-RT.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-RT.pt' --normalized \
#     --data_type 'final' --wandb --const_par 0.2 --fbnm_par 4.0 --cls_par 0.2 --corruption 'running_tap' \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 \
#     --wandb_name 'RT-10dB-org' --backup_weight 1 --early_stop 20

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/10.0-running_tap-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/10.0-running_tap-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-RT.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-RT.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-RT.pt' --normalized \
#     --data_type 'final' --wandb --const_par 0.0 --fbnm_par 4.0 --cls_par 0.2 --corruption 'running_tap' \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 \
#     --wandb_name 'RT-10dB-no_cst' --backup_weight 1 --early_stop 20

# python -m CoNMix.speech-commands.STDA \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/10.0-running_tap-weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/speech_commands/running_tap-bg/10.0-running_tap-strong' \
#     --batch_size 32 --test_batch_size 96 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelF.pt' \
#     --modelB_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelB.pt' \
#     --modelC_weight_path './result/speech-commands/CoNMix/pre_train/speech-commands_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'speech-commands_modelF-bg-10.0-RT.pt' \
#     --STDA_modelB_weight_file_name 'speech-commands_modelB-bg-10.0-RT.pt' \
#     --STDA_modelC_weight_file_name 'speech-commands_modelC-bg-10.0-RT.pt' --normalized \
#     --data_type 'final' --wandb --const_par 0.2 --fbnm_par 0.0 --cls_par 0.2 --corruption 'running_tap' \
#     --alpha 0.9 --initc_num 1 --dataset 'speech-commands' --cls_mode 'logsoft_ce' --lr_gamma 10 \
#     --wandb_name 'RT-10dB-no_nm' --backup_weight 1 --early_stop 20