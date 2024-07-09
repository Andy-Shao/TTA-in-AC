import argparse
import random
import numpy as np
import os
from tqdm import tqdm

import torch 
from torchaudio import transforms as a_transforms
from torchvision import transforms as v_transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from lib.toolkit import print_argparse, cal_norm
from lib.wavUtils import pad_trunc, Components
from CoNMix.lib.prepare_dataset import ExpandChannel
from lib.datasets import AudioMINST, load_datapath, load_from
import CoNMix.lib.models as models

def load_model(args: argparse.Namespace) -> tuple[nn.Module, nn.Module, nn.Module]:
    modelF = models.ViT().to(device=args.device)
    modelB = models.feat_bootleneck(type=args.classifier, feature_dim=modelF.in_features, bottleneck_dim=args.bottleneck).to(device=args.device)
    modelC = models.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).to(device=args.device)

    return modelF, modelB, modelC

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--temporary_path', type=str)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--modelF_weight_path', type=str)
    ap.add_argument('--modelB_weight_path', type=str)
    ap.add_argument('--modelC_weight_path', type=str)

    ap.add_argument('--corruption', type=str, default='gaussian_noise')
    ap.add_argument('--severity_level', type=float, default=.0025)

    ap.add_argument('--seed', type=int, default=2024, help='random seed')
    ap.add_argument('--cal_norm', type=str, default='none', choices=['original', 'corrupted', 'none'])

    ap.add_argument('--max_epoch', type=int, default=100, help="max iterations")
    ap.add_argument('--interval', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=48, help="batch_size")
    ap.add_argument('--test_batch_size', type=int, default=128, help="batch_size")
    ap.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    ap.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")

    ap.add_argument('--gent', type=bool, default=False)
    ap.add_argument('--ent', type=bool, default=False)
    ap.add_argument('--kd', type=bool, default=False)
    ap.add_argument('--se', type=bool, default=False)
    ap.add_argument('--nl', type=bool, default=False)
    ap.add_argument('--consist', type=bool, default=True)
    ap.add_argument('--fbnm', type=bool, default=True)

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
    ap.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    ap.add_argument('--output', type=str, default='STDA_weights', help='Save ur weights here')
    ap.add_argument('--input_source', type=str, default='pre_train', help='Load SRC training wt path')
    ap.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    ap.add_argument('--issave', type=bool, default=False)
    ap.add_argument('--earlystop', type=int, default=0)
    ap.add_argument('--plr', type=int, default=1, help='Pseudo-label refinement')
    ap.add_argument('--soft_pl', type=int, default=1)
    ap.add_argument('--suffix', type=str, default='')
    ap.add_argument('--worker', type=int, default=8)
    ap.add_argument('--wandb', type=int, default=1)
    ap.add_argument('--sdlr', type=int, default=1)

    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'CoNMix', 'analysis')
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
    #####################################

    if args.dataset == 'audio-mnist':
        max_ms=1000
        sample_rate=48000
        n_mels=128
        hop_length=377
        args.class_num = 10
        test_tf = Components(transforms=[
            pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((256, 256), antialias=False),
            v_transforms.RandomCrop(224),
            v_transforms.Normalize(mean=[-51.259285, -51.259285, -51.259285], std=[19.166618, 19.166618, 19.166618])
        ])
        audio_minst_load_pathes = load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent'] != 'German')
        test_dataset = AudioMINST(data_trainsforms=test_tf, include_rate=False, data_paths=audio_minst_load_pathes)
        
        corrupted_test_tf = v_transforms.Normalize(mean=[-30.83114, -30.83114, -30.83114], std=[7.7577553, 7.7577553, 7.7577553])
        corrupted_test_dataset = load_from(root_path=args.temporary_path, index_file_name='audio_minst_meta.csv', data_tf=corrupted_test_tf)
    else:
        raise Exception('No support')
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    corrupted_test_loader = DataLoader(dataset=corrupted_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if args.cal_norm != 'none':
        if args.cal_norm == 'original':
            mean, std = cal_norm(test_loader)
        if args.cal_norm == 'corrupted':
            mean, std = cal_norm(corrupted_test_loader)
        print(f'mean is: {mean}, std is: {std}')
        exit()
    
    print('Original Test')