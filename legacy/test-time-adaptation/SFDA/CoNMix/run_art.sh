python pre_train.py --dataset office-home --model vit --max_epoch 20 --interval 20 --batch_size 64 --source 0 --wandb 1 --output pre_train --trte full
python STDA.py --source 0 --target 1 2 3 --dataset office-home --net vit --max_epoch 50 --interval 50 --batch_size 32
python3 bridge_MTDA.py --source 0 --dataset office-home --net vit --batch_size 64
python3 MTDA.py --dataset office-home --source 0 --batch_size 64 --epoch 100 --interval 5 --net vit --wandb 1
