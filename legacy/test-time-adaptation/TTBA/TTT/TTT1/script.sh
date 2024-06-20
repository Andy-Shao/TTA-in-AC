export PYTHONPATH=$PYTHONPATH:$(pwd)

# CUDA_VISIBLE_DEVICES=0 python main.py --shared layer2 --rotation_type expand \
# 			--group_norm 8 \
# 			--nepoch 150 --milestone_1 75 --milestone_2 125 \
# 			--outf results/cifar10_layer2_gn_expand \
# 			--batch_size 256

# CUDA_VISIBLE_DEVICES=0 python script_test_c10.py --level 5 --shared layer2 --setting slow --name gn_expand
# CUDA_VISIBLE_DEVICES=0 python script_test_c10.py --level 5 --shared layer2 --setting online --name gn_expand

# CUDA_VISIBLE_DEVICES=0 python script_test_c10.py --level 4 --shared layer2 --setting slow --name gn_expand
# CUDA_VISIBLE_DEVICES=0 python script_test_c10.py --level 4 --shared layer2 --setting online --name gn_expand

# CUDA_VISIBLE_DEVICES=0 python script_test_c10.py --level 3 --shared layer2 --setting slow --name gn_expand
# CUDA_VISIBLE_DEVICES=0 python script_test_c10.py --level 3 --shared layer2 --setting online --name gn_expand

# CUDA_VISIBLE_DEVICES=0 python script_test_c10.py --level 2 --shared layer2 --setting slow --name gn_expand
# CUDA_VISIBLE_DEVICES=0 python script_test_c10.py --level 2 --shared layer2 --setting online --name gn_expand

# CUDA_VISIBLE_DEVICES=0 python script_test_c10.py --level 1 --shared layer2 --setting slow --name gn_expand
# CUDA_VISIBLE_DEVICES=0 python script_test_c10.py --level 1 --shared layer2 --setting online --name gn_expand

# CUDA_VISIBLE_DEVICES=0 python script_test_c10.py --level 0 --shared layer2 --setting slow --name gn_expand
# CUDA_VISIBLE_DEVICES=0 python script_test_c10.py --level 0 --shared layer2 --setting online --name gn_expand

# CUDA_VISIBLE_DEVICES=0 python main.py --shared layer2 --rotation_type expand \
# 			--nepoch 75 --milestone_1 50 --milestone_2 65 \
# 			--outf results/cifar10_layer2_bn_expand

# CUDA_VISIBLE_DEVICES=0 python script_test_c10.py --level 5 --shared layer2 --setting slow --name bn_expand
# CUDA_VISIBLE_DEVICES=0 python script_test_c10.py --level 5 --shared layer2 --setting online --name bn_expand

CUDA_VISIBLE_DEVICES=0 python main.py --shared layer2 --rotation_type expand \
			--nepoch 25 --milestone_1 50 --milestone_2 65 \
			--outf results/cifar10_layer2_bn_expand

CUDA_VISIBLE_DEVICES=0 python script_test_c10.py --level 5 --shared layer2 --setting slow --name bn_expand --corruption zoom_blur --dset_size 1000
# CUDA_VISIBLE_DEVICES=0 python script_test_c10.py --level 5 --shared layer2 --setting online --name bn_expand --corruption zoom_blur