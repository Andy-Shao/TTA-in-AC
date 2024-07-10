export PYTHONPATH=$PYTHONPATH:$(pwd)

# python CoNMix/analysis.py --dataset 'audio-mnist' --dataset_root_path '/root/data/AudioMNIST/data' \
#     --temporary_path '/root/tmp/AudioMNIST_analysis_005' --batch_size 256 --cal_norm 'corrupted' \
#     --severity_level 0.005 --corrupted_mean '-25.915203, -25.915203, -25.915203' \
#     --corrupted_std '6.707415, 6.707415, 6.707415'

python CoNMix/analysis.py --dataset 'audio-mnist' --dataset_root_path '/root/data/AudioMNIST/data' \
    --temporary_path '/root/tmp/AudioMNIST_analysis_005' --batch_size 64 --severity_level 0.005 \
    --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
    --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
    --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
    --output_csv_name 'accuracy_record_005.csv' --corrupted_mean '-25.911991, -25.911991, -25.911991' \
    --corrupted_std '6.705804, 6.705804, 6.705804'

# python CoNMix/analysis.py --dataset 'audio-mnist' --dataset_root_path '/root/data/AudioMNIST/data' \
#     --temporary_path '/root/tmp/AudioMNIST_analysis_005' --batch_size 64 --severity_level 0.0025 \
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --output_csv_name 'accuracy_record_0025.csv' --corrupted_mean '-30.829575, -30.829575, -30.829575' \
#     --corrupted_std '7.756144, 7.756144, 7.756144'