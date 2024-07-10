export PYTHONPATH=$PYTHONPATH:$(pwd)

python CoNMix/STDA.py --dataset 'audio-mnist' --low_aug_dataset_root_path '/root/tmp/AudioMNIST_analysis_005_low' \
    --strong_aug_dataset_root_path '/root/tmp/AudioMNIST_analysis_005_high' \
    --batch_size 32 --weak_corrupted_mean '0, 0, 0' --weak_corrupted_std '1, 1, 1' \
    --strong_corrupted_mean '0, 0, 0' --strong_corrupted_std '1, 1, 1' --cal_norm

# python CoNMix/STDA.py --dataset 'audio-mnist' --low_aug_dataset_root_path '/root/tmp/AudioMNIST_analysis_005_low' \
#     --strong_aug_dataset_root_path '/root/tmp/AudioMNIST_analysis_005_high' \
#     --batch_size 32 --severity_level 0.005 --max_epoch 5 --interval 50 \
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --weak_corrupted_mean '-25.908154, -25.908154, -25.908154' --weak_corrupted_std '6.705514, 6.705514, 6.705514' \
#     --strong_corrupted_mean '' --strong_corrupted_std '' \
#     --STDA_modelF_weight_file_name 'audio-mnist_modelF_005.pt' \
#     --STDA_modelB_weight_file_name 'audio-mnist_modelB_005.pt' \
#     --STDA_modelC_weight_file_name 'audio-mnist_modelC_005.pt'
    