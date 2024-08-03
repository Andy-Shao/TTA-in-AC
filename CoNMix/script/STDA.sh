# export PYTHONPATH=$PYTHONPATH:$(pwd)

# python -m CoNMix.STDA --dataset 'audio-mnist' --weak_aug_dataset_root_path '/home/andyshao/tmp/AudioMNIST_analysis_005_weak' \
#     --strong_aug_dataset_root_path '/home/andyshao/tmp/AudioMNIST_analysis_005_strong' \
#     --batch_size 256 --weak_corrupted_mean '0, 0, 0' --weak_corrupted_std '1, 1, 1' \
#     --strong_corrupted_mean '0, 0, 0' --strong_corrupted_std '1, 1, 1' --cal_norm

python -m CoNMix.STDA --dataset 'audio-mnist' --weak_aug_dataset_root_path '/home/andyshao/tmp/AudioMNIST_analysis_005_weak' \
    --strong_aug_dataset_root_path '/home/andyshao/tmp/AudioMNIST_analysis_005_strong' \
    --batch_size 32 --severity_level 0.005 --max_epoch 50 --interval 50 --lr '5e-4'\
    --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
    --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
    --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
    --weak_corrupted_mean '-25.919159, -25.919159, -25.919159' --weak_corrupted_std '6.7116675, 6.7116675, 6.7116675' \
    --strong_corrupted_mean '-30.639536, -30.639536, -30.639536' --strong_corrupted_std '7.445794, 7.445794, 7.445794' \
    --STDA_modelF_weight_file_name 'audio-mnist_modelF_005.pt' \
    --STDA_modelB_weight_file_name 'audio-mnist_modelB_005.pt' \
    --STDA_modelC_weight_file_name 'audio-mnist_modelC_005.pt' --wandb

# python CoNMix/STDA.py --dataset 'audio-mnist' --weak_aug_dataset_root_path '/root/tmp/AudioMNIST_analysis_0025_weak' \
#     --strong_aug_dataset_root_path '/root/tmp/AudioMNIST_analysis_0025_strong' \
#     --batch_size 256 --weak_corrupted_mean '0, 0, 0' --weak_corrupted_std '1, 1, 1' \
#     --strong_corrupted_mean '0, 0, 0' --strong_corrupted_std '1, 1, 1' --cal_norm

# python CoNMix/STDA.py --dataset 'audio-mnist' --weak_aug_dataset_root_path '/root/tmp/AudioMNIST_analysis_0025_weak' \
#     --strong_aug_dataset_root_path '/root/tmp/AudioMNIST_analysis_0025_strong' \
#     --batch_size 32 --severity_level 0.0025 --max_epoch 50 --interval 50 --lr '5e-4'\
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --weak_corrupted_mean '-30.82497, -30.82497, -30.82497' --weak_corrupted_std '7.73306, 7.73306, 7.73306' \
#     --strong_corrupted_mean '-35.46575, -35.46575, -35.46575' --strong_corrupted_std '8.521461, 8.521461, 8.521461' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF_0025.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB_0025.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC_0025.pt' --wandb