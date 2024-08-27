import argparse
import os
import pandas as pd
from tqdm import tqdm

import torch 
from torchaudio import transforms as a_transforms
from torchvision import transforms as v_transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from lib.toolkit import print_argparse, count_ttl_params, cal_norm
from lib.scDataset import BackgroundNoiseDataset, SpeechCommandsDataset
from lib.wavUtils import Components, pad_trunc, BackgroundNoise, GuassianNoise
from lib.models import WavClassifier, ElasticRestNet50
from lib.tentAdapt import get_params, TentAdapt
from lib.normAdapt import NormAdapt
from CoNMix.lib.prepare_dataset import ExpandChannel
from tent.speech_commands.pre_train import prep_dataset

def find_noise(noise_type: str) -> torch.Tensor:
    background_noise_dataset = BackgroundNoiseDataset(root_path=args.dataset_root_path)
    for _noise_type, noise, _ in background_noise_dataset:
        if noise_type == _noise_type:
            return noise
    return None

def cal_normalize(tf_array: list, args: argparse.Namespace) -> tuple:
    batch_size = 256
    data_tf = Components(transforms=tf_array)
    # target_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='test', include_rate=False, data_tfs=data_tf, data_type=args.data_type)
    target_dataset = prep_dataset(args=args, mode='test', data_tsf=data_tf, data_type=args.data_type)
    target_loader = DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_mean, test_std = cal_norm(loader=target_loader)
    return test_mean, test_std

def inference(model: nn.Module, args: argparse.Namespace, data_loader: DataLoader) -> float:
    ttl_corr = 0
    ttl_num = 0
    for features, labels in tqdm(data_loader):
        features, labels = features.to(args.device), labels.to(args.device)
        outputs = model(features)
        _, preds = torch.max(outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_num += labels.shape[0]
    ttl_accu = ttl_corr / ttl_num * 100.
    return ttl_accu

def load_model(args: argparse.Namespace) -> nn.Module:
    if args.model == 'cnn':
        model = WavClassifier(class_num=30, l1_in_features=64, c1_in_channels=1)
    elif args.model == 'restnet50':
        model = ElasticRestNet50(class_num=args.class_num)
    else:
        raise Exception('No support')
    model_stat = torch.load(f=args.model_weight_file_path)
    model.load_state_dict(model_stat)
    model = model.to(device=args.device)
    return model

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='SHOT')
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands', 'speech-commands-numbers', 'speech-commands-random'])
    ap.add_argument('--model', type=str, default='cnn', choices=['cnn', 'restnet50'])
    ap.add_argument('--model_weight_file_path', type=str)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--severity_level', default=1., type=float)
    ap.add_argument('--output_csv_name', type=str, default='accuracy_record.csv')
    ap.add_argument('--corruptions', type=str)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--tent_batch_size', type=int, default=200)
    ap.add_argument('--normalized', action='store_true')

    args = ap.parse_args()
    args.corruption_types = ['doing_the_dishes', 'dude_miaowing', 'exercise_bike', 'pink_noise', 'running_tap', 'white_noise', 'gaussian_noise']
    args.corruptions = args.corruptions.strip().split(sep=',')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dataset == 'speech-commands':
        args.class_num = 30
        args.data_type = 'all'
    elif args.dataset == 'speech-commands-numbers':
        args.class_num = 10
        args.data_type = 'numbers'
    elif args.dataset == 'speech-commands-random':
        args.class_num = 30
        args.data_type = 'all'
    else:
        raise Exception('No support')
    args.output_full_path = os.path.join(args.output_path, args.dataset, 'tent', 'analysis')
    try:
        os.makedirs(args.output_full_path)
    except:
        pass

    print_argparse(args=args)

    sample_rate = 16000
    if args.model == 'cnn':
        n_mels = 64
        data_transforms = Components(transforms=[
            pad_trunc(max_ms=1000, sample_rate=sample_rate),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels),
            a_transforms.AmplitudeToDB(top_db=80),
        ])
        # test_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='test', include_rate=False, data_tfs=data_transforms, data_type=args.data_type)
        test_dataset = prep_dataset(args=args, mode='test', data_type=args.data_type, data_tsf=data_transforms)
    elif args.model == 'restnet50':
        n_mels=129
        hop_length=125
        tf_array = [
            pad_trunc(max_ms=1000, sample_rate=sample_rate),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((224, 224), antialias=False),
        ]
        if args.normalized:
            print('calculate mean and standard deviation')
            test_mean, test_std = cal_normalize(tf_array=tf_array, args=args)
            tf_array.append(v_transforms.Normalize(mean=test_mean, std=test_std))
        test_tf = Components(transforms=tf_array)
        # test_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='test', include_rate=False, data_tfs=test_tf, data_type=args.data_type)
        test_dataset = prep_dataset(args=args, mode='test', data_tsf=test_tf, data_type=args.data_type)
    else: 
        raise Exception('No support')
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f'test data size: {len(test_dataset)}, batch size: {len(test_loader)}')

    records = pd.DataFrame(columns=['dataset', 'algorithm', 'tta-operation', 'corruption', 'accuracy', 'error', 'severity level', 'number of weight'])

    print('Original test')
    model = load_model(args)
    number_of_weight = count_ttl_params(model=model)
    model.eval()
    test_accuracy = inference(model=model, data_loader=test_loader, args=args)
    print(f'Original test accuracy: {test_accuracy:.2f}%')
    records.loc[len(records)] = [args.dataset, args.model, 'N/A', 'N/A', test_accuracy, 100. - test_accuracy, 0., number_of_weight]

    for noise_type in args.corruptions:
        assert noise_type in args.corruption_types, 'No support'
        if args.model == 'cnn':
            if noise_type == 'gaussian_noise':
                tf_array = [
                    pad_trunc(max_ms=1000, sample_rate=sample_rate),
                    GuassianNoise(noise_level=args.severity_level),
                ]
            else:
                noise = find_noise(noise_type)
                tf_array = [
                    pad_trunc(max_ms=1000, sample_rate=sample_rate),
                    BackgroundNoise(noise_level=args.severity_level, noise=noise, is_random=True),
                ]
            tf_array.extend([
                a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels),
                a_transforms.AmplitudeToDB(top_db=80),
            ])
            corrupted_tf = Components(transforms=tf_array)
        elif args.model == 'restnet50':
            if noise_type == 'gaussian_noise':
                tf_array = [
                    pad_trunc(max_ms=1000, sample_rate=sample_rate),
                    GuassianNoise(noise_level=args.severity_level),
                ]
            else:
                noise = find_noise(noise_type)
                tf_array = [
                    pad_trunc(max_ms=1000, sample_rate=sample_rate),
                    BackgroundNoise(noise_level=args.severity_level, noise=noise, is_random=True),
                ]
            tf_array.extend([
                a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
                a_transforms.AmplitudeToDB(top_db=80),
                ExpandChannel(out_channel=3),
                v_transforms.Resize((224, 224), antialias=False),
                v_transforms.RandomCrop(224),
            ])
            if args.normalized:
                print('calculate mean and standard deviation')
                test_mean, test_std = cal_normalize(tf_array=tf_array, args=args)
                tf_array.append(v_transforms.Normalize(mean=test_mean, std=test_std))
            corrupted_tf = Components(transforms=tf_array)
        # corrupted_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='test', include_rate=False, data_tfs=corrupted_tf, data_type=args.data_type)
        corrupted_dataset = prep_dataset(args=args, data_type=args.data_type, mode='test', data_tsf=corrupted_tf)
        print(f'corrupted dataset size: {len(corrupted_dataset)}')
        
        print(f'{noise_type} corruption test')
        model = load_model(args)
        number_of_weight = count_ttl_params(model=model)
        corrupted_loader = DataLoader(dataset=corrupted_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        model.eval()
        test_accuracy = inference(model=model, data_loader=corrupted_loader, args=args)
        print(f'{noise_type} corruption test accuracy: {test_accuracy:.2f}%')
        records.loc[len(records)] = [args.dataset, args.model, 'N/A', noise_type, test_accuracy, 100.-test_accuracy, args.severity_level, number_of_weight]

        print(f'{noise_type} tent test')
        model = load_model(args)
        number_of_weight = count_ttl_params(model=model)
        corrupted_loader = DataLoader(dataset=corrupted_dataset, batch_size=args.tent_batch_size, shuffle=False, drop_last=False)
        tent_params, tent_param_names = get_params(model=model)
        tent_optimizer = optim.Adam(params=tent_params, lr=1e-3, betas=(.9,.99), weight_decay=0.)
        # tent_optimizer = optim.SGD(params=model.parameters(), lr=1e-3, weight_decay=5e-4, momentum=.9)
        tent_model = TentAdapt(model=model, optimizer=tent_optimizer, steps=1, resetable=False).to(device=args.device)
        test_accuracy = inference(model=tent_model, data_loader=corrupted_loader, args=args)
        print(f'{noise_type} tent adaptation test accuracy: {test_accuracy:.2f}%')
        records.loc[len(records)] = [args.dataset, args.model, 'Tent Adaptation' if not args.normalized else 'Tent Adaptation + normalized', noise_type, test_accuracy, 100.-test_accuracy, args.severity_level, number_of_weight]
        model = None
        tent_model = None
        tent_params = None
        tent_param_names = None
        torch.cuda.empty_cache()

        print(f'{noise_type} norm test')
        if args.model == 'restnet50':
            corrupted_loader = DataLoader(dataset=corrupted_dataset, batch_size=120, shuffle=False, drop_last=False)
        model = load_model(args)
        number_of_weight = count_ttl_params(model=model)
        norm_model = NormAdapt(model=model, momentum=.9).to(device=args.device)
        test_accuracy = inference(model=norm_model, data_loader=corrupted_loader, args=args)
        print(f'{noise_type} norm adaptation test accuracy:{test_accuracy:.2f}%')
        records.loc[len(records)] = [args.dataset, args.model, 'Norm Adaptation' if not args.normalized else 'Norm Adaptation + normalized', noise_type, test_accuracy, 100. - test_accuracy, args.severity_level, number_of_weight]

    records.to_csv(os.path.join(args.output_full_path, args.output_csv_name))