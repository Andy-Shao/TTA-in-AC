import argparse
import os
import pandas as pd

import torch 
import torch.nn as nn

from lib.toolkit import print_argparse
from ttt.lib.test_helpers import build_mnist_model, time_shift_inference as inference
from ttt.lib.prepare_dataset import prepare_test_data, test_transforms

def load_model(args: argparse.Namespace, mode:str) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    assert mode in ['origin', 'adapted']
    net, ext, head, ssh = build_mnist_model(args)
    if mode == 'origin':
        stat = torch.load(args.origin_model_weight_file_path)
    elif mode == 'adapted':
        stat = torch.load(args.adapted_model_weight_file_path)
    net.load_state_dict(stat['net'])
    head.load_state_dict(stat['head'])
    return net, ext, head, ssh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    parser.add_argument('--origin_model_weight_file_path', type=str)
    parser.add_argument('--adapted_model_weight_file_path', type=str)
    parser.add_argument('--output_path', type=str, default='./result')
    parser.add_argument('--dataset_root_path', type=str)
    parser.add_argument('--rotation_type', default='rand')
    parser.add_argument('--group_norm', default=0, type=int)
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--severity_level', default=.0025, type=float)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--shared', type=str, default='layer2')

    args = parser.parse_args()
    args.output_full_path = os.path.join(args.output_path, args.dataset, 'ttt', 'analysis')
    try:
        os.makedirs(args.output_full_path)
    except:
        pass

    accu_record = pd.DataFrame(columns=['dataset', 'algorithm', 'tta-operation', 'corruption', 'accuracy', 'error', 'severity level'])
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'audio-mnist':
        args.class_num = 10
        args.sample_rate = 48000
        args.n_mels = 64
        args.final_full_line_in = 384
        args.hop_length = 505
        args.shift_limit = .25
        args.ssh_class_num = 3
        net, ext, head, ssh = load_model(args, mode='origin')
    else:
        raise Exception('No support')
    print_argparse(args)
    # Finish args prepare
    
    print('Origin test')
    args.corruption = 'original'
    test_dataset, test_loader = prepare_test_data(args=args)
    test_transf = test_transforms(args)
    original_test_accu = inference(model=net, loader=test_loader, test_transf=test_transf, device=args.device)
    accu_record.loc[len(accu_record)] = [args.dataset, 'RestNet', 'N/A', 'N/A', original_test_accu, 100. - original_test_accu, 0.]
    print(f'original data size: {len(test_dataset)}, original accuracy: {original_test_accu:.2f}%')

    print('Corruption test')
    args.corruption = 'gaussian_noise'
    corrupted_test_transf = test_transforms(args)
    corrupted_test_accu = inference(model=net, loader=test_loader, test_transf=corrupted_test_transf, device=args.device)
    accu_record.loc[len(accu_record)] = [args.dataset, 'RestNet', 'N/A', args.corruption, corrupted_test_accu, 100. - corrupted_test_accu, args.severity_level]
    print(f'corrupted data size: {len(test_dataset)}, corrupted accuracy: {corrupted_test_accu:.2f}%')

    accu_record.to_csv(os.path.join(args.output_full_path, 'accuracy_record.csv'))