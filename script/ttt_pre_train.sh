export PYTHONPATH=$PYTHONPATH:$(pwd)

python ttt/pre_time_shift_train.py --dataset_root_path '/root/data/AudioMNIST/data' --max_epoch 10 --depth 20 --shift_limit 0.2625
