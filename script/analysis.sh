export PYTHONPATH=$PYTHONPATH:$(pwd)

#python analysis.py --model_weight_file_path './result/audio-mnist/cnn/pre_train/model_weights.pt' --dataset_root_path '/root/data/AudioMNIST/data'

python ttt/time_shift_analysis.py --origin_model_weight_file_path './result/audio-mnist/ttt/pre_time_shift_train/ckpt_low.pth' \
     --dataset_root_path '/root/data/AudioMNIST/data' --depth 20