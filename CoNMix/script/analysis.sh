# export PYTHONPATH=$PYTHONPATH:$(pwd)
export BASE_PATH='/root'

python -m CoNMix.analysis --dataset 'audio-mnist' --dataset_root_path '/root/data/AudioMNIST/data' \
    --temporary_path '/root/tmp/AudioMNIST_analysis_005_weak' --batch_size 64 --severity_level 0.005 \
    --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
    --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
    --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
    --output_csv_name 'accuracy_record_005.csv' --normalized \
    --adapted_modelF_weight_path './result/audio-mnist/CoNMix/STDA/audio-mnist_modelF_005.pt' \
    --adapted_modelB_weight_path './result/audio-mnist/CoNMix/STDA/audio-mnist_modelB_005.pt' \
    --adapted_modelC_weight_path './result/audio-mnist/CoNMix/STDA/audio-mnist_modelC_005.pt'

# python CoNMix/analysis.py --dataset 'audio-mnist' --dataset_root_path '/root/data/AudioMNIST/data' \
#     --temporary_path '/root/tmp/AudioMNIST_analysis_0025_weak' --batch_size 256 --cal_norm \
#     --severity_level 0.0025

# python CoNMix/analysis.py --dataset 'audio-mnist' --dataset_root_path '/root/data/AudioMNIST/data' \
#     --temporary_path '/root/tmp/AudioMNIST_analysis_0025_weak' --batch_size 64 --severity_level 0.0025 \
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --output_csv_name 'accuracy_record_0025.csv' --corrupted_mean '-30.824112, -30.824112, -30.824112' \
#     --corrupted_std '7.765734, 7.765734, 7.765734' \
#     --test_mean '-51.259285, -51.259285, -51.259285' --test_std '19.166618, 19.166618, 19.166618' \
#     --adapted_modelF_weight_path './result/audio-mnist/CoNMix/STDA/audio-mnist_modelF_0025.pt' \
#     --adapted_modelB_weight_path './result/audio-mnist/CoNMix/STDA/audio-mnist_modelB_0025.pt' \
#     --adapted_modelC_weight_path './result/audio-mnist/CoNMix/STDA/audio-mnist_modelC_0025.pt'