#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/10.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/10.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-RT-10.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-RT-10.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-RT-10.0-bg-rand.pt' --normalized \
#     --corruption 'running_tap' --cls_par 0.0 --lr_gamma 10 --fbnm_par 4.0 --cls_mode 'logsoft_ce' \
#     --plr 1 --const_par 0.2 --wandb_name 'RT-10dB-no_pl' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/10.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/10.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-RT-10.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-RT-10.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-RT-10.0-bg-rand.pt' --normalized \
#     --corruption 'running_tap' --cls_par 1.0 --lr_gamma 10 --fbnm_par 4.0 --cls_mode 'logsoft_ce' \
#     --plr 1 --const_par 0.0 --wandb_name 'RT-10dB-no_cst' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/10.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/10.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-RT-10.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-RT-10.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-RT-10.0-bg-rand.pt' --normalized \
#     --corruption 'running_tap' --cls_par 1.0 --lr_gamma 10 --fbnm_par 0.0 --cls_mode 'logsoft_ce' \
#     --plr 1 --const_par 0.2 --wandb_name 'RT-10dB-no_nm' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/10.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/10.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-RT-10.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-RT-10.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-RT-10.0-bg-rand.pt' --normalized \
#     --corruption 'running_tap' --cls_par 0.2 --lr_gamma 30 --fbnm_par 6.0 --cls_mode 'logsoft_nll' \
#     --plr 1 --const_par 0.2 --wandb_name 'RT-10dB-upd' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/10.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/10.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-EB-10.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-EB-10.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-EB-10.0-bg-rand.pt' --normalized \
#     --corruption 'exercise_bike' --cls_par 0.0 --lr_gamma 10 --fbnm_par 4.0 --cls_mode 'logsoft_ce' \
#     --plr 1 --const_par 0.2 --wandb_name 'EB-10dB-no_pl' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/10.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/10.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-EB-10.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-EB-10.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-EB-10.0-bg-rand.pt' --normalized \
#     --corruption 'exercise_bike' --cls_par 1.0 --lr_gamma 10 --fbnm_par 4.0 --cls_mode 'logsoft_ce' \
#     --plr 1 --const_par 0.0 --wandb_name 'EB-10dB-no_cst' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/10.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/10.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-EB-10.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-EB-10.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-EB-10.0-bg-rand.pt' --normalized \
#     --corruption 'exercise_bike' --cls_par 1.0 --lr_gamma 10 --fbnm_par 0.0 --cls_mode 'logsoft_ce' \
#     --plr 1 --const_par 0.2 --wandb_name 'EB-10dB-no_nm' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/10.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/10.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-EB-10.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-EB-10.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-EB-10.0-bg-rand.pt' --normalized \
#     --corruption 'exercise_bike' --cls_par 0.2 --lr_gamma 30 --fbnm_par 6.0 --cls_mode 'logsoft_nll' \
#     --plr 1 --const_par 0.2 --wandb_name 'EB-10dB-upd' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/10.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/10.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-DD-10.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-DD-10.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-DD-10.0-bg-rand.pt' --normalized \
#     --corruption 'doing_the_dishes' --cls_par 0.0 --lr_gamma 10 --fbnm_par 4.0 --cls_mode 'logsoft_ce' \
#     --plr 1 --const_par 0.2 --wandb_name 'DD-10dB-no_pl' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/10.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/10.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-DD-10.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-DD-10.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-DD-10.0-bg-rand.pt' --normalized \
#     --corruption 'doing_the_dishes' --cls_par 1.0 --lr_gamma 10 --fbnm_par 4.0 --cls_mode 'logsoft_ce' \
#     --plr 1 --const_par 0.0 --wandb_name 'DD-10dB-no_cst' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/10.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/10.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-DD-10.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-DD-10.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-DD-10.0-bg-rand.pt' --normalized \
#     --corruption 'doing_the_dishes' --cls_par 1.0 --lr_gamma 10 --fbnm_par 0.0 --cls_mode 'logsoft_ce' \
#     --plr 1 --const_par 0.2 --wandb_name 'DD-10dB-no_nm' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/10.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/10.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-DD-10.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-DD-10.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-DD-10.0-bg-rand.pt' --normalized \
#     --corruption 'doing_the_dishes' --cls_par 0.2 --lr_gamma 10 --fbnm_par 6.0 --cls_mode 'logsoft_nll' \
#     --plr 1 --const_par 0.2 --wandb_name 'DD-10dB-upd' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/3.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/3.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-RT-3.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-RT-3.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-RT-3.0-bg-rand.pt' --normalized \
#     --corruption 'running_tap' --cls_par 1.0 --lr_gamma 10 --fbnm_par 4.0 --cls_mode 'logsoft_ce' \
#     --plr 0 --const_par 0.2 --wandb_name 'RT-3dB-org' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/3.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/3.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-RT-3.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-RT-3.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-RT-3.0-bg-rand.pt' --normalized \
#     --corruption 'running_tap' --cls_par 1.0 --lr_gamma 10 --fbnm_par 4.0 --cls_mode 'logsoft_ce' \
#     --plr 0 --const_par 0.0 --wandb_name 'RT-3dB-no_cst' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/3.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/3.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-RT-3.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-RT-3.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-RT-3.0-bg-rand.pt' --normalized \
#     --corruption 'running_tap' --cls_par 1.0 --lr_gamma 10 --fbnm_par 0.0 --cls_mode 'logsoft_ce' \
#     --plr 0 --const_par 0.2 --wandb_name 'RT-3dB-no_nm' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/3.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/3.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-RT-3.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-RT-3.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-RT-3.0-bg-rand.pt' --normalized \
#     --corruption 'running_tap' --cls_par 0.2 --lr_gamma 30 --fbnm_par 6.0 --cls_mode 'logsoft_nll' \
#     --plr 0 --const_par 0.2 --wandb_name 'RT-3dB-upd' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/3.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/3.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-EB-3.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-EB-3.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-EB-3.0-bg-rand.pt' --normalized \
#     --corruption 'exercise_bike' --cls_par 1.0 --lr_gamma 10 --fbnm_par 4.0 --cls_mode 'logsoft_ce' \
#     --plr 0 --const_par 0.2 --wandb_name 'EB-3dB-org' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/3.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/3.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-EB-3.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-EB-3.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-EB-3.0-bg-rand.pt' --normalized \
#     --corruption 'exercise_bike' --cls_par 1.0 --lr_gamma 10 --fbnm_par 4.0 --cls_mode 'logsoft_ce' \
#     --plr 0 --const_par 0.0 --wandb_name 'EB-3dB-no_cst' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/3.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/3.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-EB-3.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-EB-3.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-EB-3.0-bg-rand.pt' --normalized \
#     --corruption 'exercise_bike' --cls_par 1.0 --lr_gamma 10 --fbnm_par 0.0 --cls_mode 'logsoft_ce' \
#     --plr 0 --const_par 0.2 --wandb_name 'EB-3dB-no_nm' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/3.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/3.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-EB-3.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-EB-3.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-EB-3.0-bg-rand.pt' --normalized \
#     --corruption 'exercise_bike' --cls_par 0.2 --lr_gamma 30 --fbnm_par 6.0 --cls_mode 'logsoft_nll' \
#     --plr 0 --const_par 0.2 --wandb_name 'EB-3dB-upd' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/3.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/3.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-DD-3.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-DD-3.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-DD-3.0-bg-rand.pt' --normalized \
#     --corruption 'doing_the_dishes' --cls_par 1.0 --lr_gamma 10 --fbnm_par 4.0 --cls_mode 'logsoft_ce' \
#     --plr 0 --const_par 0.2 --wandb_name 'DD-3dB-org' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/3.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/3.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-DD-3.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-DD-3.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-DD-3.0-bg-rand.pt' --normalized \
#     --corruption 'doing_the_dishes' --cls_par 1.0 --lr_gamma 10 --fbnm_par 4.0 --cls_mode 'logsoft_ce' \
#     --plr 0 --const_par 0.0 --wandb_name 'DD-3dB-no_cst' --backup_weight 1 --early_stop 20 --wandb

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/3.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/3.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-DD-3.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-DD-3.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-DD-3.0-bg-rand.pt' --normalized \
#     --corruption 'doing_the_dishes' --cls_par 1.0 --lr_gamma 10 --fbnm_par 0.0 --cls_mode 'logsoft_ce' \
#     --plr 0 --const_par 0.2 --wandb_name 'DD-3dB-no_nm' --backup_weight 1 --early_stop 20 --wandb

python -m CoNMix.STDA --dataset 'audio-mnist' \
    --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/3.0-bg-rand_weak' \
    --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/3.0-bg-rand_strong' \
    --batch_size 32 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
    --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
    --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
    --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
    --STDA_modelF_weight_file_name 'audio-mnist_modelF-DD-3.0-bg-rand.pt' \
    --STDA_modelB_weight_file_name 'audio-mnist_modelB-DD-3.0-bg-rand.pt' \
    --STDA_modelC_weight_file_name 'audio-mnist_modelC-DD-3.0-bg-rand.pt' --normalized \
    --corruption 'doing_the_dishes' --cls_par 0.2 --lr_gamma 30 --fbnm_par 6.0 --cls_mode 'logsoft_nll' \
    --plr 0 --const_par 0.2 --wandb_name 'DD-3dB-upd' --backup_weight 1 --early_stop 20 --wandb