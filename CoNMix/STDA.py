import argparse
import wandb
import os
import random
import numpy as np
from tqdm import tqdm

import torch 
from torchvision import transforms as v_transforms
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from lib.toolkit import print_argparse, parse_mean_std, cal_norm
from lib.datasets import load_from
from CoNMix.analysis import load_model, load_origin_stat
from CoNMix.pre_train import op_copy
from CoNMix.lib.prepare_dataset import Dataset_Idx, ExpandChannel
from lib.wavUtils import Components, time_shift, pad_trunc

def build_optim(args: argparse.Namespace, modelF: nn.Module, modelB: nn.Module, modelC: nn.Module) -> optim.Optimizer:
    param_group = []
    for _, v in modelF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params':v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for _, v in modelB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params':v, 'lr': args.lr * args.lr_decay2}]
        else: 
            v.requires_grad = False
    for _, v in modelC.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params':v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(params=param_group)
    optimizer = op_copy(optimizer)
    return optimizer

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Rand-Augment')
    ap.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    ap.add_argument('--weak_aug_dataset_root_path', type=str)
    ap.add_argument('--strong_aug_dataset_root_path', type=str)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--modelF_weight_path', type=str)
    ap.add_argument('--modelB_weight_path', type=str)
    ap.add_argument('--modelC_weight_path', type=str)
    ap.add_argument('--STDA_modelF_weight_file_name', type=str)
    ap.add_argument('--STDA_modelB_weight_file_name', type=str)
    ap.add_argument('--STDA_modelC_weight_file_name', type=str)
    ap.add_argument('--wandb', action='store_true')

    ap.add_argument('--corruption', type=str, default='gaussian_noise')
    ap.add_argument('--severity_level', type=float, default=.0025)
    ap.add_argument('--weak_corrupted_mean', type=str)
    ap.add_argument('--weak_corrupted_std', type=str)
    ap.add_argument('--strong_corrupted_mean', type=str)
    ap.add_argument('--strong_corrupted_std', type=str)

    ap.add_argument('--seed', type=int, default=2024, help='random seed')
    ap.add_argument('--cal_norm', type=str, default='none', choices=['original', 'corrupted', 'none'])

    ap.add_argument('--max_epoch', type=int, default=100, help="max iterations")
    ap.add_argument('--interval', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=48, help="batch_size")
    # ap.add_argument('--test_batch_size', type=int, default=128, help="batch_size")
    ap.add_argument('--lr', type=float, default=1e-3, help="learning rate")

    # ap.add_argument('--gent', type=bool, default=False)
    # ap.add_argument('--ent', type=bool, default=False)
    # ap.add_argument('--kd', type=bool, default=False)
    # ap.add_argument('--se', type=bool, default=False)
    # ap.add_argument('--nl', type=bool, default=False)
    # ap.add_argument('--consist', type=bool, default=True)
    # ap.add_argument('--fbnm', type=bool, default=True)

    ap.add_argument('--threshold', type=int, default=0)
    ap.add_argument('--cls_par', type=float, default=0.2, help='lambda 2')
    ap.add_argument('--alpha', type=float, default=0.9)

    ap.add_argument('--const_par', type=float, default=0.2, help='lambda 3')
    ap.add_argument('--ent_par', type=float, default=1.3)
    ap.add_argument('--fbnm_par', type=float, default=4.0, help='lambda 1')

    ap.add_argument('--lr_decay1', type=float, default=0.1)
    ap.add_argument('--lr_decay2', type=float, default=1.0)

    ap.add_argument('--bottleneck', type=int, default=256)
    ap.add_argument('--epsilon', type=float, default=1e-5)
    ap.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    ap.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    # ap.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    # ap.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    # ap.add_argument('--issave', type=bool, default=False)
    # ap.add_argument('--earlystop', type=int, default=0)
    # ap.add_argument('--plr', type=int, default=1, help='Pseudo-label refinement')
    # ap.add_argument('--soft_pl', type=int, default=1)
    # ap.add_argument('--suffix', type=str, default='')
    # ap.add_argument('--worker', type=int, default=8)
    # ap.add_argument('--sdlr', type=int, default=1)

    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'CoNMix', 'STDA')
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
    #################################################

    if args.dataset == 'audio-mnist':
        max_ms=1000
        sample_rate=48000
        n_mels=128
        hop_length=377
        args.class_num = 10
        weak_test_tf = Components(transforms=[
            pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
            time_shift(shift_limit=.25, is_random=True, is_bidirection=True),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((256, 256), antialias=False),
            v_transforms.RandomCrop(224),
            v_transforms.RandomHorizontalFlip(), 
            v_transforms.Normalize(mean=parse_mean_std(args.weak_corrupted_mean), std=parse_mean_std(args.weak_corrupted_std))
        ])
        weak_test_dataset = load_from(root_path=args.weak_aug_dataset_root_path, index_file_name='audio_minst_meta.csv', data_tf=low_test_tf)
        weak_test_dataset = Dataset_Idx(dataset=weak_test_dataset)
        
        strong_test_tf = Components(transforms=[
            pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
            time_shift(shift_limit=.25, is_random=True, is_bidirection=True),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((256, 256), antialias=False),
            v_transforms.RandomCrop(224),
            v_transforms.RandomHorizontalFlip(), 
            v_transforms.Normalize(mean=parse_mean_std(args.strong_corrupted_mean), std=parse_mean_std(args.strong_corrupted_std))
        ])
        strong_test_dataset = load_from(root_path=args.strong_aug_dataset_root_path, index_file_name='audio_minst_meta.csv', data_tf=strong_test_tf)
    else:
        raise Exception('No support')
    
    weak_test_loader = DataLoader(dataset=weak_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if args.cal_norm != 'none':
        if args.cal_norm == 'original':
            mean, std = cal_norm(loader=weak_test_loader)
        if args.cal_norm == 'corrupted':
            mean, std = cal_norm(loader=DataLoader(dataset=strong_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False))
        print(f'mean: {mean}, std: {std}')
        exit()

    wandb_run = wandb.init(
        project='Audio Classification STDA (CoNMix)', name=args.dataset, mode='online' if args.wandb else 'disabled',
        config=args, tags=['Audio Classification', args.dataset, 'ViT'])

    # build mode & load pre-train weight
    modelF, modelB, modelC = load_model(args)
    load_origin_stat(args, modelF=modelF, modelB=modelB, modelC=modelC)
    
    optimizer = build_optim(args, modelF=modelF, modelB=modelB, modelC=modelC)
    max_iter = args.max_epoch * len(weak_test_loader)
    interval_iter = max_iter // args.interval
    iter = 0

    print('STDA Training Started')