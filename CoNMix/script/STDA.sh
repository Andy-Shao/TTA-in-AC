export PYTHONPATH=$PYTHONPATH:$(pwd)

# python CoNMix/STDA.py --dataset 'audio-mnist' --weak_aug_dataset_root_path '/root/tmp/AudioMNIST_analysis_005_weak' \
#     --strong_aug_dataset_root_path '/root/tmp/AudioMNIST_analysis_005_strong' \
#     --batch_size 256 --weak_corrupted_mean '0, 0, 0' --weak_corrupted_std '1, 1, 1' \
#     --strong_corrupted_mean '0, 0, 0' --strong_corrupted_std '1, 1, 1' --cal_norm

python CoNMix/STDA.py --dataset 'audio-mnist' --weak_aug_dataset_root_path '/root/tmp/AudioMNIST_analysis_005_weak' \
    --strong_aug_dataset_root_path '/root/tmp/AudioMNIST_analysis_005_strong' \
    --batch_size 32 --severity_level 0.005 --max_epoch 10 --interval 10 \
    --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
    --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
    --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
    --weak_corrupted_mean '-25.912006, -25.912006, -25.912006' --weak_corrupted_std '6.7070265, 6.7070265, 6.7070265' \
    --strong_corrupted_mean '-30.647058, -30.647058, -30.647058' --strong_corrupted_std '7.4446855, 7.4446855, 7.4446855' \
    --STDA_modelF_weight_file_name 'audio-mnist_modelF_005.pt' \
    --STDA_modelB_weight_file_name 'audio-mnist_modelB_005.pt' \
    --STDA_modelC_weight_file_name 'audio-mnist_modelC_005.pt' --wandb