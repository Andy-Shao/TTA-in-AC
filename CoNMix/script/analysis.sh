# export PYTHONPATH=$PYTHONPATH:$(pwd)
export BASE_PATH='/root'

# python -m CoNMix.analysis --dataset 'audio-mnist' --dataset_root_path '/root/data/AudioMNIST/data' \
#     --temporary_path '/root/tmp/AudioMNIST_analysis_005_weak' --batch_size 64 --severity_level 0.005 \
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --output_csv_name 'accuracy_record_005.csv' --normalized \
#     --adapted_modelF_weight_path './result/audio-mnist/CoNMix/STDA/audio-mnist_modelF_005.pt' \
#     --adapted_modelB_weight_path './result/audio-mnist/CoNMix/STDA/audio-mnist_modelB_005.pt' \
#     --adapted_modelC_weight_path './result/audio-mnist/CoNMix/STDA/audio-mnist_modelC_005.pt'

# python -m CoNMix.analysis --dataset 'audio-mnist' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --temporary_path $BASE_PATH'/tmp/AudioMNIST_analysis_0025_weak' --batch_size 64 --severity_level 0.0025 \
#     --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
#     --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
#     --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
#     --output_csv_name 'accuracy_record_0025.csv' --normalized \
#     --adapted_modelF_weight_path './result/audio-mnist/CoNMix/STDA/audio-mnist_modelF_0025.pt' \
#     --adapted_modelB_weight_path './result/audio-mnist/CoNMix/STDA/audio-mnist_modelB_0025.pt' \
#     --adapted_modelC_weight_path './result/audio-mnist/CoNMix/STDA/audio-mnist_modelC_0025.pt'

python -m CoNMix.analysis --dataset 'audio-mnist' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --temporary_path $BASE_PATH'/tmp/AudioMNIST_analysis/doing_the_dishes/1.0-bg-rand_weak' \
    --batch_size 64 --severity_level 1.0 \
    --modelF_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelF.pt' \
    --modelB_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelB.pt' \
    --modelC_weight_path './result/audio-mnist/CoNMix/pre_train/audio-mnist_best_modelC.pt' \
    --output_csv_name 'accuracy_record-doing_the_dishes-1.0-bg-rand.csv' --normalized \
    --adapted_modelF_weight_path './result/audio-mnist/CoNMix/STDA/audio-mnist_modelF-doing_the_dishes-1.0-bg-rand.pt' \
    --adapted_modelB_weight_path './result/audio-mnist/CoNMix/STDA/audio-mnist_modelB-doing_the_dishes-1.0-bg-rand.pt' \
    --adapted_modelC_weight_path './result/audio-mnist/CoNMix/STDA/audio-mnist_modelC-doing_the_dishes-1.0-bg-rand.pt'