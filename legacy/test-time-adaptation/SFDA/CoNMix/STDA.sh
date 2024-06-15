# python STDA.py --gpu_id 0 --s 0 --t 1 2 3 --dset office-home --net vit --max_epoch 50 --interval 50 --batch_size 32
# python STDA.py --gpu_id 0 --s 1 --t 0 2 3 --dset office-home --net vit --max_epoch 50 --interval 50 --batch_size 32
# python STDA.py --gpu_id 0 --s 2 --t 1 0 3 --dset office-home --net vit --max_epoch 50 --interval 50 --batch_size 32
# python STDA.py --gpu_id 0 --s 3 --t 1 2 0 --dset office-home --net vit --max_epoch 50 --interval 50 --batch_size 32

python STDA.py --source 0 --target 1 2 3 --dataset office-home --net vit --max_epoch 50 --interval 10 --batch_size 32
python STDA.py --source 1 --target 0 2 3 --dataset office-home --net vit --max_epoch 50 --interval 10 --batch_size 32
python STDA.py --source 2 --target 1 0 3 --dataset office-home --net vit --max_epoch 50 --interval 10 --batch_size 32
python STDA.py --source 3 --target 1 2 0 --dataset office-home --net vit --max_epoch 50 --interval 10 --batch_size 32
