# export PYTHONPATH=$PYTHONPATH:$(pwd)
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m CoNMix.STDA --dataset 'audio-mnist' --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis_005_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis_005_strong' \
#     --batch_size 32 --severity_level 0.005 --max_epoch 50 --interval 50 --lr '5e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF_005.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB_005.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC_005.pt' --wandb --normalized

# python -m CoNMix.STDA --dataset 'audio-mnist' --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis_0025_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis_0025_strong' \
#     --batch_size 32 --severity_level 0.0025 --max_epoch 50 --interval 50 --lr '5e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF_0025.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB_0025.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC_0025.pt' --wandb --normalized

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/3.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/3.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-doing_the_dishes-3.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-doing_the_dishes-3.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-doing_the_dishes-3.0-bg-rand.pt' --wandb --normalized \
#     --corruption 'doing_the_dishes' --cls_par 0.0 --lr_gamma 30 --fbnm_par 4.0 --cls_mode 'logsoft_nll' \
#     --plr 0 --const_par 0.6

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/3.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/3.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-exercise_bike-3.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-exercise_bike-3.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-exercise_bike-3.0-bg-rand.pt' --wandb --normalized \
#     --corruption 'exercise_bike' --cls_par 0.0 --lr_gamma 30 --fbnm_par 4.0 --cls_mode 'logsoft_nll' \
#     --plr 0 --const_par 0.6

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/3.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/3.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 3.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-running_tap-3.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-running_tap-3.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-running_tap-3.0-bg-rand.pt' --wandb --normalized \
#     --corruption 'running_tap' --cls_par 0.0 --lr_gamma 30 --fbnm_par 4.0 --cls_mode 'logsoft_nll' \
#     --plr 0 --const_par 0.6

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/10.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/10.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-doing_the_dishes-10.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-doing_the_dishes-10.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-doing_the_dishes-10.0-bg-rand.pt' --wandb --normalized \
#     --corruption 'doing_the_dishes' --cls_par 1.0 --lr_gamma 10 --fbnm_par 4.0 --cls_mode 'logsoft_ce' \
#     --plr 1 --const_par 0.2

# python -m CoNMix.STDA --dataset 'audio-mnist' \
#     --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/10.0-bg-rand_weak' \
#     --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/exercise_bike/10.0-bg-rand_strong' \
#     --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF-exercise_bike-10.0-bg-rand.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB-exercise_bike-10.0-bg-rand.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC-exercise_bike-10.0-bg-rand.pt' --wandb --normalized \
#     --corruption 'exercise_bike' --cls_par 1.0 --lr_gamma 10 --fbnm_par 4.0 --cls_mode 'logsoft_ce' \
#     --plr 1 --const_par 0.2

python -m CoNMix.STDA --dataset 'audio-mnist' \
    --weak_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/10.0-bg-rand_weak' \
    --strong_aug_dataset_root_path $BASE_PATH'/tmp/AudioMNIST_analysis/running_tap/10.0-bg-rand_strong' \
    --batch_size 32 --severity_level 10.0 --max_epoch 50 --interval 50 --lr '1e-4'\
    --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
    --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
    --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
    --STDA_modelF_weight_file_name 'audio-mnist_modelF-running_tap-10.0-bg-rand.pt' \
    --STDA_modelB_weight_file_name 'audio-mnist_modelB-running_tap-10.0-bg-rand.pt' \
    --STDA_modelC_weight_file_name 'audio-mnist_modelC-running_tap-10.0-bg-rand.pt' --wandb --normalized \
    --corruption 'running_tap' --cls_par 1.0 --lr_gamma 10 --fbnm_par 4.0 --cls_mode 'logsoft_ce' \
    --plr 1 --const_par 0.2