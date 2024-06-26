import argparse
import os
from tqdm import tqdm
import pandas as pd

import torch 
import torch.nn as nn
import torchaudio.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from lib.normAdapt import NormAdapt
from lib.tentAdapt import TentAdapt, get_params
from lib.datasets import AudioMINST, load_datapath
from lib.wavUtils import Components, pad_trunc, GuassianNoise
from lib.models import WavClassifier
from lib.toolkit import print_argparse

def inference(model: nn.Module, loader: DataLoader, device: str) -> float:
    test_accu = 0.
    test_size = 0.
    test_iter = iter(loader)
    for iteration in tqdm(range(len(loader))):
        inputs, labels = next(test_iter)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)
        test_accu += (preds == labels).sum().cpu().item()
        test_size += labels.shape[0]
    return test_accu / test_size * 100.

def load_model(args: argparse.Namespace, device='cuda') -> nn.Module:
    if args.model == 'cnn':
        model = WavClassifier(class_num=10, l1_in_features=64, c1_in_channels=1).to(device=device)
        model.load_state_dict(torch.load(args.model_weight_file_path))
    else:
        raise Exception('Cannot be satisified the requirment')
    return model

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='SHOT')
    ap.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    ap.add_argument('--model', type=str, default='cnn', choices=['cnn'])
    ap.add_argument('--model_weight_file_path', type=str)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--dataset_root_path', type=str)

    args = ap.parse_args()
    args.severity_level = .0025
    args.corruption = 'gaussian_noise'
    print_argparse(args=args)
    args.output_full_path = os.path.join(args.output_path, args.dataset, args.model, 'analysis')
    try:
        os.makedirs(args.output_full_path)
    except:
        pass

    accu_record = pd.DataFrame(columns=['dataset', 'algorithm', 'tta-operation', 'corruption', 'accuracy', 'error', 'severity level'])
    running_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Original test')
    if args.dataset == 'audio-mnist':
        n_mel = 64
        sample_rate = 48000
        data_transforms = Components(transforms=[
            pad_trunc(max_ms=1000, sample_rate=sample_rate),
            transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mel),
            transforms.AmplitudeToDB(top_db=80)
        ])
        test_datapathes = load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent']!= 'German')
        test_dataset = AudioMINST(data_paths=test_datapathes, data_trainsforms=data_transforms, include_rate=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, drop_last=False)
    else:
        raise Exception('No support')
    
    model = load_model(args, device=running_device)
    
    model.eval()
    original_test_accuracy = inference(model=model, loader=test_loader, device=running_device)
    print(f'original test data size: {len(test_dataset)}, original test accuracy: {original_test_accuracy:.2f}%')
    accu_record.loc[len(accu_record)] = [args.dataset, args.model, 'N/A', 'N/A', original_test_accuracy, 100. - original_test_accuracy, 0.]

    print('Corruption test')
    if args.dataset == 'audio-mnist':
        corrupted_data_transforms = Components(transforms=[
            pad_trunc(max_ms=1000, sample_rate=sample_rate),
            GuassianNoise(noise_level=args.severity_level), # .025
            transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mel),
            transforms.AmplitudeToDB(top_db=80)
        ])
        corrupted_test_dataset = AudioMINST(data_paths=test_datapathes, include_rate=False, data_trainsforms=corrupted_data_transforms)
        corrupted_test_loader = DataLoader(dataset=corrupted_test_dataset, batch_size=64, shuffle=False, drop_last=False)
    else:
        raise Exception('No support')
    model = load_model(args, device=running_device)
    model.eval()
    corrupted_test_accuracy = inference(model=model, loader=corrupted_test_loader, device=running_device)
    print(f'corrupted test data size: {len(corrupted_test_dataset)}, corrupted test accuracy: {corrupted_test_accuracy:.2f}%')
    accu_record.loc[len(accu_record)] = [args.dataset, args.model, 'N/A', args.corruption, corrupted_test_accuracy, 100. - corrupted_test_accuracy, args.severity_level]

    print('Tent Adaptation')
    if args.dataset == 'audio-mnist':
        tent_test_loader = DataLoader(dataset=corrupted_test_dataset, batch_size=200, shuffle=False, drop_last=False)
    else:
        raise Exception('No support')
    model = load_model(args, running_device)
    tent_params, tent_param_names = get_params(model=model)
    tent_optimizer = optim.Adam(params=tent_params, lr=1e-3, betas=(.9,.99), weight_decay=0.)
    tent_model = TentAdapt(model=model, optimizer=tent_optimizer, steps=1, resetable=False).to(device=running_device)
    tent_accu = inference(model=tent_model, loader=tent_test_loader, device=running_device)
    accu_record.loc[len(accu_record)] = [args.dataset, args.model, 'Tent Adaptation', args.corruption, tent_accu, 100. - tent_accu, args.severity_level]
    print(f'tent test data size: {len(corrupted_test_dataset)}, tent test accuracy:{tent_accu:.2f}%')

    print('Norm Adaptation')
    model = load_model(args, running_device)
    norm_model = NormAdapt(model=model).to(device=running_device)
    norm_accu = inference(model=norm_model, loader=tent_test_loader, device=running_device)
    accu_record.loc[len(accu_record)] = [args.dataset, args.model, 'Norm Adaptation', args.corruption, norm_accu, 100. - norm_accu, args.severity_level]
    print(f'norm test data size: {len(corrupted_test_dataset)}, norm test accuracy: {norm_accu:.2f}%')

    # Stroe the record
    accu_record.to_csv(os.path.join(args.output_full_path, 'accuracy_records.csv'))