# Office-Home
# CUDA_VISIBLE_DEVICES=1 python3 bridge_MTDA.py --s 0 --dset office-home --net deit_s --batch_size 64
# CUDA_VISIBLE_DEVICES=1 python3 bridge_MTDA.py --s 1 --dset office-home --net deit_s --batch_size 64
# CUDA_VISIBLE_DEVICES=1 python3 bridge_MTDA.py --s 2 --dset office-home --net deit_s --batch_size 64
# CUDA_VISIBLE_DEVICES=1 python3 bridge_MTDA.py --s 3 --dset office-home --net deit_s --batch_size 64

python3 bridge_MTDA.py --source 0 --dataset office-home --net vit --batch_size 64
python3 bridge_MTDA.py --source 1 --dataset office-home --net vit --batch_size 64
python3 bridge_MTDA.py --source 2 --dataset office-home --net vit --batch_size 64
python3 bridge_MTDA.py --source 3 --dataset office-home --net vit --batch_size 64
