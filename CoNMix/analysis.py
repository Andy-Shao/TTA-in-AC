import argparse
import random
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

import torch 
from torchaudio import transforms as a_transforms
from torchvision import transforms as v_transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from lib.toolkit import print_argparse, cal_norm, parse_mean_std, count_ttl_params
from lib.wavUtils import pad_trunc, Components
from CoNMix.lib.prepare_dataset import ExpandChannel
from lib.datasets import AudioMINST, load_datapath, load_from
import CoNMix.lib.models as models

def inference(modelF: nn.Module, modelB: nn.Module, modelC: nn.Module, data_loader: DataLoader, device='cpu') -> float:
    modelF.eval()
    modelB.eval()
    modelC.eval()
    ttl_corr = 0.
    ttl_size = 0.
    for features, labels in tqdm(data_loader):
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            outputs = modelC(modelB(modelF(features)))
        _, preds = torch.max(outputs, dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]
    return ttl_corr / ttl_size * 100.

def load_model(args: argparse.Namespace) -> tuple[nn.Module, nn.Module, nn.Module]:
    modelF = models.ViT().to(device=args.device)
    modelB = models.feat_bootleneck(type=args.classifier, feature_dim=modelF.in_features, bottleneck_dim=args.bottleneck).to(device=args.device)
    modelC = models.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).to(device=args.device)

    return modelF, modelB, modelC

def load_origin_stat(args: argparse.Namespace, modelF: nn.Module, modelB: nn.Module, modelC: nn.Module) -> None:
    modelF.load_state_dict(torch.load(args.modelF_weight_path))
    modelB.load_state_dict(torch.load(args.modelB_weight_path))
    modelC.load_state_dict(torch.load(args.modelC_weight_path))

def load_adapted_stat(args: argparse.Namespace, modelF: nn.Module, modelB: nn.Module, modelC: nn.Module) -> None:
    modelF.load_state_dict(torch.load(args.adapted_modelF_weight_path))
    modelB.load_state_dict(torch.load(args.adapted_modelB_weight_path))
    modelC.load_state_dict(torch.load(args.adapted_modelC_weight_path))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--temporary_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--modelF_weight_path', type=str)
    ap.add_argument('--modelB_weight_path', type=str)
    ap.add_argument('--modelC_weight_path', type=str)
    ap.add_argument('--output_csv_name', type=str, default='accuracy_record.csv')
    ap.add_argument('--adapted_modelF_weight_path', type=str)
    ap.add_argument('--adapted_modelB_weight_path', type=str)
    ap.add_argument('--adapted_modelC_weight_path', type=str)

    ap.add_argument('--corruption', type=str, default='gaussian_noise')
    ap.add_argument('--severity_level', type=float, default=.0025)

    ap.add_argument('--seed', type=int, default=2024, help='random seed')
    ap.add_argument('--normalized', action='store_true')

    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--interval', type=int, default=50, help='interval')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--bottleneck', type=int, default=256)
    ap.add_argument('--epsilon', type=float, default=1e-5)
    ap.add_argument('--layer', type=str, default='wn', choices=['linear', 'wn'])
    ap.add_argument('--classifier', type=str, default='bn', choices=['ori', 'bn'])
    # ap.add_argument('--smooth', type=float, default=.1)
    # ap.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    # ap.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    # ap.add_argument('--bsp', type=bool, default=False)
    # ap.add_argument('--se', type=bool, default=False)
    # ap.add_argument('--nl', type=bool, default=False)
    # ap.add_argument('--worker', type=int, default=16)

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
        test_tf = [
            pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((256, 256), antialias=False),
            v_transforms.RandomCrop(224),
        ]   
        audio_minst_load_pathes = load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent'] != 'German')
        if args.normalized:
            print('calculate the test dataset mean and standard deviation')
            test_dataset = AudioMINST(data_trainsforms=Components(transforms=test_tf), include_rate=False, data_paths=audio_minst_load_pathes)
            test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=args.num_workers)
            test_mean, test_std = cal_norm(loader=test_loader)
            test_tf.append(v_transforms.Normalize(mean=test_mean, std=test_std))
        test_dataset = AudioMINST(data_trainsforms=Components(transforms=test_tf), include_rate=False, data_paths=audio_minst_load_pathes)
        
        corrupted_test_tf = []
        if args.normalized:
            corrupted_test_dataset = load_from(root_path=args.temporary_path, index_file_name='audio_minst_meta.csv')
            corrupted_test_loader = DataLoader(dataset=corrupted_test_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=args.num_workers)
            corrupted_mean, corrupted_std = cal_norm(loader=corrupted_test_loader)
            corrupted_test_tf.append(v_transforms.Normalize(mean=corrupted_mean, std=corrupted_std))
        corrupted_test_dataset = load_from(root_path=args.temporary_path, index_file_name='audio_minst_meta.csv', data_tf=Components(transforms=corrupted_test_tf))
    else:
        raise Exception('No support')
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    corrupted_test_loader = DataLoader(dataset=corrupted_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    
    print('Original Test')
    modelF, modelB, modelC = load_model(args)
    ttl_weight_num = count_ttl_params(model=modelF) + count_ttl_params(model=modelB) + count_ttl_params(model=modelC)
    load_origin_stat(args, modelF=modelF, modelB=modelB, modelC=modelC)
    accuracy = inference(modelF=modelF, modelB=modelB, modelC=modelC, data_loader=test_loader, device=args.device)
    print(f'Oritinal Test -- data size is:{len(test_dataset)}, accuracy: {accuracy:.2f}%')
    accu_record.loc[len(accu_record)] = [args.dataset, args.algorithm, 'N/A', 'N/A', accuracy, 100. - accuracy, 0., ttl_weight_num]

    print('Corrupted Test')
    modelF, modelB, modelC = load_model(args)
    ttl_weight_num = count_ttl_params(model=modelF) + count_ttl_params(model=modelB) + count_ttl_params(model=modelC)
    load_origin_stat(args, modelF=modelF, modelB=modelB, modelC=modelC)
    accuracy = inference(modelF=modelF, modelB=modelB, modelC=modelC, data_loader=corrupted_test_loader, device=args.device)
    print(f'Corrupted Test -- data size is:{len(corrupted_test_dataset)}, accuracy: {accuracy:.2f}%')
    accu_record.loc[len(accu_record)] = [args.dataset, args.algorithm, 'N/A', args.corruption, accuracy, 100. - accuracy, args.severity_level, ttl_weight_num]

    print('CoNMix Test')
    modelF, modelB, modelC = load_model(args)
    ttl_weight_num = count_ttl_params(model=modelF) + count_ttl_params(model=modelB) + count_ttl_params(model=modelC)
    load_adapted_stat(args, modelF=modelF, modelB=modelB, modelC=modelC)
    accuracy = inference(modelF=modelF, modelB=modelB, modelC=modelC, data_loader=corrupted_test_loader, device=args.device)
    print(f'CoNMix Test -- data size is:{len(corrupted_test_dataset)}, accuracy:{accuracy:.2f}%')
    accu_record.loc[len(accu_record)] = [args.dataset, args.algorithm, 'CoNMix', args.corruption, accuracy, 100. - accuracy, args.severity_level, ttl_weight_num]

    accu_record.to_csv(os.path.join(args.full_output_path, args.output_csv_name))