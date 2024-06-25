import argparse
import os

import torch.nn as nn

from lib.toolkit import print_argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--max_epoch', type=int, default=75)
    parser.add_argument('--lr', type=float, default=.1)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--dataset_root_path', type=str)
    parser.add_argument('--output_path', type=str, default='./result')
    parser.add_argument('--shared', type=str, default='layer2')
    parser.add_argument('--milestone_1', default=50, type=int)
    parser.add_argument('--milestone_2', default=65, type=int)
    parser.add_argument('--rotation_type', default='rand')
    parser.add_argument('--group_norm', default=0, type=int)
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)

    args = parser.parse_args()
    print('TTT pre-train')
    print_argparse(args=args)
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    args.output_full_path = os.path.join(args.output_path, args.dataset, 'ttt', 'pre_train')
    try:
        os.makedirs(args.output_full_path)
    except:
        pass

    if args.dataset == 'audio-mnist':
        args.class_num = 10
    else:
        raise Exception('No support')