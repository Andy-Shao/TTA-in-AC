import argparse
import os
import random
import numpy as np

import torch 

from lib.toolkit import print_argparse, count_ttl_params
from ttt.lib.test_helpers import build_sc_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands'])
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
    parser.add_argument('--shift_limit', default=.25, type=float)
    parser.add_argument('--output_csv_name', type=str, default='accu_record.csv')
    parser.add_argument('--output_weight_name', type=str, default='ckpt.pth')
    parser.add_argument('--seed', default=2024, type=int)

    args = parser.parse_args()
    print('TTT pre-train')
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    args.output_full_path = os.path.join(args.output_path, args.dataset, 'ttt', 'pre_time_shift_train')
    try:
        os.makedirs(args.output_full_path)
    except:
        pass

    if args.dataset == 'speech-commands':
        args.class_num = 10
        args.sample_rate = 16000
        args.n_mels = 64
        args.final_full_line_in = 384
        args.hop_length = 253
        args.ssh_class_num = 3
    else:
        raise Exception('No support')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print_argparse(args=args)
    # Finished args prepare
################################################################

    net, ext, head, ssh = build_sc_model(args=args)
    print(f'net weight number is: {count_ttl_params(net)}, ssh weight number is: {count_ttl_params(ssh)}, ext weight number is: {count_ttl_params(ext)}')
    print((f'total weight number is: {count_ttl_params(net) + count_ttl_params(head)}'))