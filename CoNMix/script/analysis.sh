export PYTHONPATH=$PYTHONPATH:$(pwd)

# python CoNMix/analysis.py --dataset 'audio-mnist' --dataset_root_path '/root/data/AudioMNIST/data' \
#     --temporary_path '/root/tmp/AudioMNIST_analysis_0025' --batch_size 256 --cal_norm 'original' \
#     --severity_level 0.0025

python CoNMix/analysis.py --dataset 'audio-mnist' --dataset_root_path '/root/data/AudioMNIST/data' \
    --temporary_path '/root/tmp/AudioMNIST_analysis_0025' --batch_size 256 --severity_level 0.0025 \
    --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
    --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
    --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt'