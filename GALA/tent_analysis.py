import argparse
import os
from tqdm import tqdm
import pandas as pd

import torch 
import torch.nn as nn
import torchaudio.transforms as a_transforms
import torchvision.transforms as v_transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from lib.tentAdapt import get_params, model_formate
from lib.datasets import AudioMINST, load_datapath
from lib.wavUtils import Components, pad_trunc, BackgroundNoise
from lib.models import ElasticRestNet50
from lib.toolkit import print_argparse, count_ttl_params, cal_norm
from CoNMix.lib.prepare_dataset import ExpandChannel
from lib.scDataset import BackgroundNoiseDataset
from GALA.lib.galaTentAdapt import adapted_forward

def find_noise(args:argparse.Namespace, new_sample_rate: int, corruption:str) -> torch.Tensor:
    data_tsf = [
        a_transforms.Resample(orig_freq=16000, new_freq=new_sample_rate)
    ]
    backgrounds = BackgroundNoiseDataset(root_path=args.background_root_path, data_tf=Components(transforms=data_tsf), label_tf=lambda x: new_sample_rate)
    for noise_type, noise, _ in backgrounds:
        if corruption == noise_type:
            return noise
    return None

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

        inputs = None
        labels = None
        outputs = None
        preds = None
    return test_accu / test_size * 100.

def gala_inference(
        model: nn.Module, loader: DataLoader, device: str, optimizer: optim.Optimizer, lr:float,
        threshold:float, step=1
    ) -> float:
    test_accu = 0.
    test_size = 0.
    for index in tqdm(range(len(loader) * step)):
        if index % len(loader) == 0:
            test_iter = iter(loader)
        inputs, labels = next(test_iter)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = adapted_forward(
            x=inputs, model=model, optimizer=optimizer, lr=lr, threshold=threshold, 
            selected=False if index % len(loader) == 0 else True
        )
        if index >= (step - 1) * len(loader):
            _, preds = torch.max(outputs.detach(), dim=1)
            test_accu += (preds == labels).sum().cpu().item()
            test_size += labels.shape[0]

        inputs = None
        labels = None
        outputs = None
        preds = None
    return test_accu / test_size * 100.

def load_model(args: argparse.Namespace, device='cuda') -> nn.Module:
    if args.model == 'restnet50':
        model = ElasticRestNet50(class_num=10).to(device=device)
        model.load_state_dict(torch.load(args.model_weight_file_path))
    else:
        raise Exception('Cannot be satisified the requirment')
    return model

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--model', type=str, default='restnet50', choices=['restnet50'])
    ap.add_argument('--model_weight_file_path', type=str)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--background_root_path', type=str)
    ap.add_argument('--severity_level', default=.0025, type=float)
    ap.add_argument('--output_csv_name', type=str, default='accuracy_record.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--tent_batch_size', type=int, default=200)
    ap.add_argument('--normalized', action='store_true')
    ap.add_argument('--corruptions', type=str)
    ap.add_argument('--rand_bg', action='store_true')
    ap.add_argument('--threshold', type=float, default=.75)
    ap.add_argument('--step', type=int, default=1)

    args = ap.parse_args()
    args.corruptions = [it.strip() for it in args.corruptions.strip().split(',')]
    args.corruption_types = ['doing_the_dishes', 'dude_miaowing', 'exercise_bike', 'pink_noise', 'running_tap', 'white_noise']
    print_argparse(args=args)
    args.output_full_path = os.path.join(args.output_path, args.dataset, 'GALA', 'analysis')
    try:
        os.makedirs(args.output_full_path)
    except:
        pass

    accu_record = pd.DataFrame(columns=['dataset', 'algorithm', 'tta-operation', 'corruption', 'accuracy', 'error', 'severity level', 'number of weight'])
    running_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'audio-mnist' and args.model == 'restnet50':
        sample_rate = 48000
        n_mels=128
        hop_length=377
        test_datapathes = load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent']!= 'German')
        test_tf = [
            pad_trunc(max_ms=1000, sample_rate=sample_rate),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((224, 224), antialias=False),
        ]
        if args.normalized:
            print('calculate the test dataset mean and standard deviation')
            test_dataset = AudioMINST(data_paths=test_datapathes, data_trainsforms=Components(transforms=test_tf), include_rate=False)
            test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, drop_last=True, num_workers=args.num_workers)
            test_mean, test_std = cal_norm(loader=test_loader)
            test_tf.append(v_transforms.Normalize(mean=test_mean, std=test_std))
        test_dataset = AudioMINST(data_paths=test_datapathes, data_trainsforms=Components(transforms=test_tf), include_rate=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    else:
        raise Exception('No support')
    
    print('Original test')
    model = load_model(args, device=running_device)
    weight_num = count_ttl_params(model=model)
    model.eval()
    original_test_accuracy = inference(model=model, loader=test_loader, device=running_device)
    print(f'original test data size: {len(test_dataset)}, original test accuracy: {original_test_accuracy:.2f}%')
    accu_record.loc[len(accu_record)] = [args.dataset, args.model, 'N/A' if not args.normalized else 'normalized', 'N/A', original_test_accuracy, 100. - original_test_accuracy, 0., weight_num]

    for corruption in args.corruptions:
        assert corruption in args.corruption_types, 'No support'
        noise = find_noise(args=args, new_sample_rate=sample_rate, corruption=corruption)
        corruption_ops = f'{corruption}-rand' if args.rand_bg else f'{corruption}-fixed'
        if args.dataset == 'audio-mnist' and args.model == 'restnet50':
            corrupted_test_tf =[
                pad_trunc(max_ms=1000, sample_rate=sample_rate),
                BackgroundNoise(noise=noise, noise_level=args.severity_level, is_random=args.rand_bg),
                a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels),
                a_transforms.AmplitudeToDB(top_db=80),
                ExpandChannel(out_channel=3),
                v_transforms.Resize((256, 256), antialias=False),
                v_transforms.RandomCrop(224),
            ]
            if args.normalized:
                print('calculate the corrupted dataset mean and standard deviation')
                corrupted_test_dataset = AudioMINST(data_paths=test_datapathes, include_rate=False, data_trainsforms=Components(transforms=corrupted_test_tf))
                corrupted_test_loader = DataLoader(dataset=corrupted_test_dataset, batch_size=256, shuffle=False, drop_last=True, num_workers=args.num_workers)
                corrupted_mean, corrupted_std = cal_norm(loader=corrupted_test_loader)
                corrupted_test_tf.append(v_transforms.Normalize(mean=corrupted_mean, std=corrupted_std))
            corrupted_test_dataset = AudioMINST(data_paths=test_datapathes, include_rate=False, data_trainsforms=Components(transforms=corrupted_test_tf))
            corrupted_test_loader = DataLoader(dataset=corrupted_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

        print(f'{corruption} Corruption test')
        model = load_model(args, device=running_device)
        weight_num = count_ttl_params(model=model)
        model.eval()
        corrupted_test_accuracy = inference(model=model, loader=corrupted_test_loader, device=running_device)
        print(f'{corruption} corrupted test data size: {len(corrupted_test_dataset)}, corrupted test accuracy: {corrupted_test_accuracy:.2f}%')
        accu_record.loc[len(accu_record)] = [args.dataset, args.model, 'N/A' if not args.normalized else 'normalized', corruption_ops, corrupted_test_accuracy, 100. - corrupted_test_accuracy, args.severity_level, weight_num]

        print(f'{corruption} GALA+Tent Adaptation')
        if args.dataset == 'audio-mnist':
            tent_test_loader = DataLoader(dataset=corrupted_test_dataset, batch_size=args.tent_batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
        else:
            raise Exception('No support')
        model = load_model(args, running_device).to(device=running_device)
        weight_num = count_ttl_params(model=model)
        tent_params, tent_param_names = get_params(model=model)
        tent_optimizer = optim.Adam(params=tent_params, lr=1e-3, betas=(.9,.99), weight_decay=0.)
        tent_model = model_formate(model=model)
        tent_accu = gala_inference(
            model=tent_model, loader=tent_test_loader, device=running_device, lr=1e-3, threshold=args.threshold,
            optimizer=tent_optimizer, step=args.step
        )
        accu_record.loc[len(accu_record)] = [args.dataset, args.model, 'GALA+Tent Adaptation' if not args.normalized else 'GALA+Tent Adaptation + normalized', corruption_ops, tent_accu, 100. - tent_accu, args.severity_level, weight_num]
        print(f'gala+tent test data size: {len(corrupted_test_dataset)}, gala+tent test accuracy:{tent_accu:.2f}%')
        tent_model = None
        tent_optimizer = None
        model = None
        tent_params = None
        tent_param_names = None

    # Store the record
    accu_record.to_csv(os.path.join(args.output_full_path, args.output_csv_name))