export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=0 python main.py --shared layer2 --rotation_type expand \
			--group_norm 8 \
			--nepoch 2 --milestone_1 75 --milestone_2 125 \
			--outf results/cifar10_layer2_gn_expand