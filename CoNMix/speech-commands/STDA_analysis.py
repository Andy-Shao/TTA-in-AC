import argparse
import os
import numpy as np
import random
import pandas as pd

import torch 
from torchaudio import transforms as a_transforms
from torchvision import transforms as v_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse, cal_norm, count_ttl_params
from lib.wavUtils import Components, pad_trunc, DoNothing
from CoNMix.lib.prepare_dataset import ExpandChannel
from lib.scDataset import SpeechCommandsDataset
from lib.datasets import load_from
from CoNMix.analysis import load_model, load_origin_stat, inference, load_adapted_stat

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands', 'speech-commands-purity'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num-workers', type=int, default=16)
    ap.add_argument('--temporary_path', type=str)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--modelF_weight_path', type=str)
    ap.add_argument('--modelB_weight_path', type=str)
    ap.add_argument('--modelC_weight_path', type=str)
    ap.add_argument('--output_csv_name', type=str, default='accuracy_record.csv')
    ap.add_argument('--adapted_modelF_weight_path', type=str)
    ap.add_argument('--adapted_modelB_weight_path', type=str)
    ap.add_argument('--adapted_modelC_weight_path', type=str)

    ap.add_argument('--corruption', type=str, choices=['doing_the_dishes', 'dude_miaowing', 'exercise_bike', 'pink_noise', 'running_tap', 'white_noise', 'guassian_noise'])
    ap.add_argument('--severity_level', type=float, default=.0025)
    ap.add_argument('--data_type', type=str, choices=['raw', 'final'], default='final')

    ap.add_argument('--seed', type=int, default=2024, help='random seed')
    ap.add_argument('--normalized', action='store_true')
    ap.add_argument('--analyze_STDA', action='store_true')

    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--interval', type=int, default=50, help='interval')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--bottleneck', type=int, default=256)
    ap.add_argument('--epsilon', type=float, default=1e-5)
    ap.add_argument('--layer', type=str, default='wn', choices=['linear', 'wn'])
    ap.add_argument('--classifier', type=str, default='bn', choices=['ori', 'bn'])

    args = ap.parse_args()
    args.algorithm = 'ViT'
    accu_record = pd.DataFrame(columns=['dataset', 'algorithm', 'tta-operation', 'corruption', 'accuracy', 'error', 'severity level', 'number of weight'])
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'CoNMix', 'analysis')
    try:
        os.makedirs(args.full_output_path)
    except:
        pass
    torch.backends.cudnn.benchmark = True

    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    if args.dataset == 'speech-commands':
        args.class_num = 30
        args.dataset_type = 'all'
    elif args.dataset == 'speech-commands-purity':
        args.class_num = 10
        args.dataset_type = 'commands'
    else:
        raise Exception('No support')

    print_argparse(args)
    #####################################

    max_ms=1000
    sample_rate=16000
    n_mels=81
    hop_length=200
    meta_file_name = 'speech_commands_meta.csv'
    algorithm = 'R50+ViT-B_16'
    tf_array = [
        pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
        a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
        a_transforms.AmplitudeToDB(top_db=80),
        ExpandChannel(out_channel=3),
        v_transforms.Resize((224, 224), antialias=False)
    ]
    if args.normalized:
        print('calculate the test dataset mean and standard deviation')
        test_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='test', include_rate=False, data_tfs=Components(transforms=tf_array), data_type=args.dataset_type)
        test_mean, test_std = cal_norm(loader=DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=args.num_workers))
        tf_array.append(v_transforms.Normalize(mean=test_mean, std=test_std))
    test_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='test', include_rate=False, data_tfs=Components(transforms=tf_array), data_type=args.dataset_type)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    if args.data_type == 'final':
        tf_array = [DoNothing()]
    elif args.data_type == 'raw':
        tf_array = [
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((224, 224), antialias=False)
        ]
    if args.normalized:
        print('calculate the corrupted test dataset mean and standard deviation')
        corrupted_dataset = load_from(root_path=args.temporary_path, index_file_name=meta_file_name, data_tf=Components(transforms=tf_array))
        corr_mean, corr_std = cal_norm(loader=DataLoader(dataset=corrupted_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=args.num_workers))
        tf_array.append(v_transforms.Normalize(mean=corr_mean, std=corr_std))
    corrupted_dataset = load_from(root_path=args.temporary_path, index_file_name=meta_file_name, data_tf=Components(transforms=tf_array))
    corrupted_loader = DataLoader(dataset=corrupted_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    print('Original test')
    modelF, modelB, modelC = load_model(args)
    ttl_weight_num = count_ttl_params(model=modelF) + count_ttl_params(model=modelB) + count_ttl_params(model=modelC)
    load_origin_stat(args, modelF=modelF, modelB=modelB, modelC=modelC)
    accuracy = inference(modelF=modelF, modelB=modelB, modelC=modelC, data_loader=test_loader, device=args.device)
    print(f'Oritinal Test -- data size is:{len(test_dataset)}, accuracy: {accuracy:.2f}%')
    accu_record.loc[len(accu_record)] = [args.dataset, algorithm, 'N/A', 'N/A', accuracy, 100. - accuracy, 0., ttl_weight_num]

    print('Corrupted Test')
    modelF, modelB, modelC = load_model(args)
    ttl_weight_num = count_ttl_params(model=modelF) + count_ttl_params(model=modelB) + count_ttl_params(model=modelC)
    load_origin_stat(args, modelF=modelF, modelB=modelB, modelC=modelC)
    accuracy = inference(modelF=modelF, modelB=modelB, modelC=modelC, data_loader=corrupted_loader, device=args.device)
    print(f'Corrupted Test -- data size is:{len(corrupted_loader)}, accuracy: {accuracy:.2f}%')
    accu_record.loc[len(accu_record)] = [args.dataset, algorithm, 'N/A', args.corruption, accuracy, 100. - accuracy, args.severity_level, ttl_weight_num]

    if args.analyze_STDA:
        print('STDA analysis')
        modelF, modelB, modelC = load_model(args)
        ttl_weight_num = count_ttl_params(model=modelF) + count_ttl_params(model=modelB) + count_ttl_params(model=modelC)
        load_adapted_stat(args, modelF=modelF, modelB=modelB, modelC=modelC)
        accuracy = inference(modelF=modelF, modelB=modelB, modelC=modelC, data_loader=corrupted_loader, device=args.device)
        print(f'STDA analysis -- data size is: {len(corrupted_loader)}, accuracy: {accuracy:.2f}%')
        accu_record.loc[len(accu_record)] = [args.dataset, algorithm, 'CoNMix-STDA', args.corruption, accuracy, 100. - accuracy, args.severity_level, ttl_weight_num]
    
    accu_record.to_csv(os.path.join(args.full_output_path, args.output_csv_name))