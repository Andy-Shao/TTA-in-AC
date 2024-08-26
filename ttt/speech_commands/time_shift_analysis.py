import argparse
import os
import pandas as pd
from tqdm import tqdm

import torch 
from torch import nn
from torch import optim

from lib.toolkit import print_argparse, count_ttl_params
from ttt.lib.test_helpers import build_sc_model, time_shift_inference as inference
from ttt.lib.speech_commands.prepare_dataset import prepare_data, val_transforms as test_transforms, train_transforms
from lib.scDataset import BackgroundNoiseDataset
from lib.wavUtils import Components, pad_trunc, GuassianNoise, BackgroundNoise
from ttt.lib.prepare_dataset import TimeShiftOps
from ttt.time_shift_analysis import adapt_one, measure_one

def find_noise(noise_type:str, args:argparse) -> torch.Tensor:
    backgrounds = BackgroundNoiseDataset(root_path=args.dataset_root_path)
    for _noise_type, noise, _ in backgrounds:
        if _noise_type == noise_type:
            return noise

def refresh_model(args: argparse.Namespace, net: nn.Module, head: nn.Module):
    stat = torch.load(args.origin_model_weight_file_path)
    net.load_state_dict(stat['net'])
    head.load_state_dict(stat['head'])

def load_model(args: argparse.Namespace) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    net, ext, head, ssh = build_sc_model(args)
    refresh_model(args=args, net=net, head=head)
    return net, ext, head, ssh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands', 'speech-commands-numbers'])
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--origin_model_weight_file_path', type=str)
    parser.add_argument('--output_path', type=str, default='./result')
    parser.add_argument('--output_csv_name', type=str, default='accuracy_record.csv')
    parser.add_argument('--dataset_root_path', type=str)
    parser.add_argument('--rotation_type', default='rand')
    parser.add_argument('--group_norm', default=0, type=int)
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--severity_level', default=.0025, type=float)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--shared', type=str, default='layer2')
    ########################################################################
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--niter', default=1, type=int)
    parser.add_argument('--threshold', default=1., type=float)
    parser.add_argument('--shift_limit', default=.25, type=float)
    parser.add_argument('--corruptions')
    parser.add_argument('--TTT_analysis', action='store_true')

    args = parser.parse_args()
    args.corruptions = [it.strip() for it in args.corruptions.strip().split(sep=',')]
    args.corruption_types = ['doing_the_dishes', 'dude_miaowing', 'exercise_bike', 'pink_noise', 'running_tap', 'white_noise', 'gaussian_noise']
    args.output_full_path = os.path.join(args.output_path, args.dataset, 'ttt', 'time_shift_analysis')
    try:
        os.makedirs(args.output_full_path)
    except:
        pass

    accu_record = pd.DataFrame(columns=['dataset', 'algorithm', 'tta-operation', 'corruption', 'accuracy', 'error', 'severity level', 'number of weight'])
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'speech-commands':
        args.class_num = 30
        args.sample_rate = 16000
        args.n_mels = 64
        args.final_full_line_in = 256
        args.hop_length = 253
        args.ssh_class_num = 3
        net, ext, head, ssh = load_model(args)
        args.data_type = 'all'
    if args.dataset == 'speech-commands-numbers':
        args.class_num = 10
        args.sample_rate = 16000
        args.n_mels = 64
        args.final_full_line_in = 256
        args.hop_length = 253
        args.ssh_class_num = 3
        net, ext, head, ssh = load_model(args)
        args.data_type = 'numbers'
    else:
        raise Exception('No support')
    
    torch.backends.cudnn.benchmark = True

    if args.group_norm == 0:
        args.ttt_operation = f'TTT, ts, bn'
    else: 
        args.ttt_operation = f'TTT, ts, gn'
    print_argparse(args)
    # Finish args prepare
    ##########################################

    print('Origin test')
    test_dataset, test_loader = prepare_data(args=args, mode='test', data_type=args.data_type)
    test_transf = test_transforms(args)
    ttl_weight_num = count_ttl_params(model=net) + count_ttl_params(model=ext) + count_ttl_params(model=head)
    test_accu = inference(model=net, loader=test_loader, test_transf=test_transf, device=args.device)
    accu_record.loc[len(accu_record)] = [args.dataset, f'RestNet{args.depth}_base', 'N/A', 'N/A', test_accu, 100. - test_accu, 0., ttl_weight_num]
    print(f'original data size: {len(test_dataset)}, original accuracy: {test_accu:.2f}%')

    for corruption in args.corruptions:
        assert corruption in args.corruption_types, 'No support'
        if corruption == 'gaussian_noise':
            tf_array = [
                pad_trunc(max_ms=1000, sample_rate=args.sample_rate),
                GuassianNoise(noise_level=args.severity_level),
            ]
        else:
            noise = find_noise(noise_type=corruption, args=args)
            tf_array = [
                pad_trunc(max_ms=1000, sample_rate=args.sample_rate),
                BackgroundNoise(noise_level=args.severity_level, noise=noise, is_random=True),
            ]
        corrupted_dataset, corrupted_loader = prepare_data(args=args, mode='test', data_transforms=Components(transforms=tf_array), data_type=args.data_type)

        print(f'{corruption} corrupted test')
        net, ext, head, ssh = load_model(args)
        ttl_weight_num = count_ttl_params(model=net) + count_ttl_params(model=ext) + count_ttl_params(model=head)
        test_accu = inference(model=net, loader=corrupted_loader, test_transf=test_transf, device=args.device)
        accu_record.loc[len(accu_record)] = [args.dataset, f'RestNet{args.depth}_base', 'N/A', corruption, test_accu, 100. - test_accu, args.severity_level, ttl_weight_num]
        print(f'corrupted data size: {len(test_dataset)}, corrupted accuracy: {test_accu:.2f}%')

        if args.TTT_analysis:

            print(f'{corruption} TTT test')
            net, ext, head, ssh = load_model(args)
            ttl_weight_num = count_ttl_params(model=net) + count_ttl_params(model=ext) + count_ttl_params(model=head)
            args.batch_size = args.batch_size // 3
            train_transfs = train_transforms(args)
            criterion_ssh = nn.CrossEntropyLoss().to(device=args.device)
            optimizer_ssh = optim.SGD(params=ssh.parameters(), lr=args.lr)
            ttt_corr = 0
            for feature, label in tqdm(corrupted_dataset):
                input = test_transf[TimeShiftOps.ORIGIN].tran_one(feature)
                input = input.to(args.device)
                _, confidence = measure_one(model=ssh, audio=input, label=0)
                if confidence < args.threshold:
                    adapt_one(feature=feature, ssh=ssh, ext=ext, args=args, criterion=criterion_ssh, data_transf=train_transfs, 
                        optimizer=optimizer_ssh, net=net, head=head, mode='online')
                correctness, confidence = measure_one(model=net, audio=input, label=label)
                ttt_corr += correctness
            ttt_accu = ttt_corr / len(test_dataset) * 100.
            print(f'Online TTT adaptation data size: {len(test_dataset)}, accuracy: {ttt_accu:.2f}%')
            accu_record.loc[len(accu_record)] = [args.dataset, f'RestNet{args.depth}_base', args.ttt_operation + ', online', corruption, ttt_accu, 100. - ttt_accu, args.severity_level, ttl_weight_num]

    accu_record.to_csv(os.path.join(args.output_full_path, args.output_csv_name))