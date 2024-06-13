# Office-Home
# CUDA_VISIBLE_DEVICES=1 python3 bridge_MTDA.py --s 0 --dset office-home --net deit_s --batch_size 64
# CUDA_VISIBLE_DEVICES=1 python3 bridge_MTDA.py --s 1 --dset office-home --net deit_s --batch_size 64
# CUDA_VISIBLE_DEVICES=1 python3 bridge_MTDA.py --s 2 --dset office-home --net deit_s --batch_size 64
# CUDA_VISIBLE_DEVICES=1 python3 bridge_MTDA.py --s 3 --dset office-home --net deit_s --batch_size 64

# python3 MTDA.py --dset office-home --s 0 --batch_size 64 --epoch 100 --interval 5 --net deit_s --gpu_id 0
# python3 MTDA.py --dset office-home --s 1 --batch_size 64 --epoch 100 --interval 5 --net deit_s --gpu_id 1
# python3 MTDA.py --dset office-home --s 2 --batch_size 64 --epoch 100 --interval 5 --net deit_s --gpu_id 1
# python3 MTDA.py --dset office-home --s 2 --batch_size 64 --epoch 100 --interval 5 --net deit_s --gpu_id 0

python3 bridge_MTDA.py --source 0 --dataset office-home --net vit --batch_size 64