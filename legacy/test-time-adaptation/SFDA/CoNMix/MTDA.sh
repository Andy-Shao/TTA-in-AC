# Office-Home
# python3 MTDA.py --dset office-home --s 0 --batch_size 64 --epoch 100 --interval 5 --net deit_s --gpu_id 0
# python3 MTDA.py --dset office-home --s 1 --batch_size 64 --epoch 100 --interval 5 --net deit_s --gpu_id 1
# python3 MTDA.py --dset office-home --s 2 --batch_size 64 --epoch 100 --interval 5 --net deit_s --gpu_id 1
# python3 MTDA.py --dset office-home --s 2 --batch_size 64 --epoch 100 --interval 5 --net deit_s --gpu_id 0

python3 MTDA.py --dataset office-home --source 0 --batch_size 64 --epoch 2 --interval 5 --net vit --wandb 1
python3 MTDA.py --dataset office-home --source 1 --batch_size 64 --epoch 2 --interval 5 --net vit --wandb 1
python3 MTDA.py --dataset office-home --source 2 --batch_size 64 --epoch 2 --interval 5 --net vit --wandb 1
python3 MTDA.py --dataset office-home --source 3 --batch_size 64 --epoch 2 --interval 5 --net vit --wandb 1