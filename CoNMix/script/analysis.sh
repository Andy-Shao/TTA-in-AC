export PYTHONPATH=$PYTHONPATH:$(pwd)

# python CoNMix/analysis.py --dataset 'audio-mnist' --dataset_root_path '/root/data/AudioMNIST/data' \
#     --temporary_path '/root/tmp/AudioMNIST_analysis' --batch_size 256 --cal_norm 'corrupted'

python CoNMix/analysis.py --dataset 'audio-mnist' --dataset_root_path '/root/data/AudioMNIST/data' \
    --temporary_path '/root/tmp/AudioMNIST_analysis' --batch_size 64