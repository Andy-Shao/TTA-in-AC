import argparse
import os
import numpy as np
import random
import wandb

import torch
import torch.nn as nn
import torchvision.transforms as v_transforms
import torchaudio.transforms as a_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse, cal_norm, store_model_structure_to_txt
from lib.wavUtils import Components, pad_trunc, time_shift
from CoNMix.lib.prepare_dataset import ExpandChannel
from lib.datasets import load_datapath, AudioMINST, ClipDataset
from AuT.lib.models import AuT, AudioClassifier

def load_models(args: argparse.Namespace) -> tuple[nn.Module, nn.Module]:
    auT = AuT().to(device=args.device)
    auC = AudioClassifier(
        type=args.classifier, feature_dim=auT.out_features, bottleneck_dim=args.bottleneck, cls_type=args.layer,
        class_num=args.class_num
    )
    auC = auC.to(device=args.device)
    return auT, auC

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_csv_name', type=str, default='training_records.csv')
    ap.add_argument('--output_weight_prefix', type=str, default='speech-commands')
    ap.add_argument('--temporary_path', type=str)
    ap.add_argument('--wandb', action='store_true')

    ap.add_argument('--seed', type=int, default=2024, help='random seed')
    ap.add_argument('--normalized', action='store_true')
    ap.add_argument('--test_rate', type=float, default=.3)

    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--interval', type=int, default=50, help='interval')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--bottleneck', type=int, default=256)
    ap.add_argument('--epsilon', type=float, default=1e-5)
    ap.add_argument('--layer', type=str, default='wn', choices=['linear', 'wn'])
    ap.add_argument('--classifier', type=str, default='bn', choices=['ori', 'bn'])
    ap.add_argument('--smooth', type=float, default=.1)
    ap.add_argument('--trte', type=str, default='val', choices=['full', 'val'])

    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'AuT', 'pre_train')
    try:
        os.makedirs(args.full_output_path)
    except:
        pass
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################

    wandb_run = wandb.init(
        project='Audio Classification Pre-Training (AuT)', name=args.dataset, mode='online' if args.wandb else 'disabled',
        config=args, tags=['Audio Classification', args.dataset, 'AuT'])
    
    if args.dataset == 'audio-mnist':
        max_ms=1000
        sample_rate=48000
        n_mels=128
        hop_length=377
        mean_vals = [-52.9362, -52.9362, -52.9362]
        std_vals = [19.3889, 19.3889, 19.3889]
        train_tf = [
            pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
            time_shift(shift_limit=.25, is_random=True, is_bidirection=True),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((256, 256), antialias=False),
            v_transforms.RandomCrop(224),
            v_transforms.RandomHorizontalFlip(),
        ]
        train_data_paths = load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent'] == 'German')
        if args.normalized:
            print('calculate mean and standard deviation')
            train_dataset = AudioMINST(data_paths=train_data_paths, data_trainsforms=Components(transforms=train_tf), include_rate=False)
            mean_vals, std_vals = cal_norm(loader=DataLoader(dataset=train_dataset, batch_size=256, shuffle=False, drop_last=False))
            train_tf.append(v_transforms.Normalize(mean=mean_vals, std=std_vals))
        train_dataset = AudioMINST(data_paths=train_data_paths, data_trainsforms=Components(transforms=train_tf), include_rate=False)
        args.class_num = 10
        test_tf = [
            pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((224, 224), antialias=False),
            v_transforms.Normalize(mean=mean_vals, std=std_vals)
        ]
        test_dataset = AudioMINST(data_paths=train_data_paths, data_trainsforms=Components(transforms=test_tf), include_rate=False)
        test_dataset = ClipDataset(dataset=test_dataset, rate=args.test_rate)
    else:
        raise Exception('No support')
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    auT, auC = load_models(args=args)
    store_model_structure_to_txt(model=auT, output_path=os.path.join(args.full_output_path, 'AuT_structure.txt'))
    store_model_structure_to_txt(model=auC, output_path=os.path.join(args.full_output_path, 'AuC_structure.txt'))