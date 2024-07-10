export PYTHONPATH=$PYTHONPATH:$(pwd)

# python CoNMix/analysis.py --dataset 'audio-mnist' --dataset_root_path '/root/data/AudioMNIST/data' \
#     --temporary_path '/root/tmp/AudioMNIST_analysis_005_low' --batch_size 256 --cal_norm 'corrupted' \
#     --severity_level 0.005 --corrupted_mean '0, 0, 0' \
#     --corrupted_std '1, 1, 1'

python CoNMix/analysis.py --dataset 'audio-mnist' --dataset_root_path '/root/data/AudioMNIST/data' \
    --temporary_path '/root/tmp/AudioMNIST_analysis_005_low' --batch_size 64 --severity_level 0.005 \
    --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
    --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
    --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
    --output_csv_name 'accuracy_record_005.csv' --corrupted_mean '-25.91517, -25.91517, -25.91517' \
    --corrupted_std '6.711713, 6.711713, 6.711713'