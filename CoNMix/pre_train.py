import argparse
import os
import random
import numpy as np
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
from torchaudio import transforms as a_transforms
from torchvision import transforms as v_transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from lib.toolkit import print_argparse, store_model_structure_to_txt
from lib.wavUtils import Components, pad_trunc, time_shift
from lib.datasets import AudioMINST, load_datapath, ClipDataset
from CoNMix.lib.prepare_dataset import ExpandChannel
import CoNMix.lib.models as models
from CoNMix.lib.loss import CrossEntropyLabelSmooth

def lr_scheduler(optimizer: torch.optim.Optimizer, iter_num: int, max_iter: int, gamma=10, power=0.75) -> optim.Optimizer:
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = .9
        param_group['nestenv'] = True
    return optimizer

def build_optimizer(args: argparse.Namespace, modelF: nn.Module, modelB: nn.Module, modelC: nn.Module) -> optim.Optimizer:
    param_group = []
    learning_rate = args.lr
    for k, v in modelF.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate * .1}]
    for k, v in modelB.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate}]
    for k, v in modelC.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    return optimizer

def op_copy(optimizer: optim.Optimizer) -> optim.Optimizer:
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def load_models(args: argparse.Namespace) -> tuple[nn.Module, nn.Module, nn.Module]:
    modelF = models.ViT().to(device=args.device)
    modelB = models.feat_bootleneck(type=args.classifier, feature_dim=modelF.in_features, bottleneck_dim=args.bottleneck).to(device=args.device)
    modelC = models.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).to(device=args.device)

    return modelF, modelB, modelC

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_csv_name', type=str, default='training_records.csv')
    ap.add_argument('--output_weight_prefix', type=str, default='audio-mnist')
    ap.add_argument('--temporary_path', type=str)
    ap.add_argument('--wandb', action='store_true')

    # ap.add_argument('--corruption', type=str, default='gaussian_noise')
    # ap.add_argument('--severity_level', type=float, default=.0025)

    ap.add_argument('--seed', type=int, default=2024, help='random seed')
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
    # ap.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    ap.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    # ap.add_argument('--bsp', type=bool, default=False)
    # ap.add_argument('--se', type=bool, default=False)
    # ap.add_argument('--nl', type=bool, default=False)
    # ap.add_argument('--worker', type=int, default=16)

    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'CoNMix', 'pre_train')
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
        project='Audio Classification Pre-Training (CoNMix)', name=args.dataset, mode='online' if args.wandb else 'disabled',
        config=args, tags=['Audio Classification', args.dataset, 'ViT'])

    if args.dataset == 'audio-mnist':
        max_ms=1000
        sample_rate=48000
        n_mels=128
        hop_length=377
        mean_vals = [-52.9362, -52.9362, -52.9362]
        std_vals = [19.3889, 19.3889, 19.3889]
        train_tf = Components(transforms=[
            pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
            time_shift(shift_limit=.25, is_random=True, is_bidirection=True),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((256, 256), antialias=False),
            v_transforms.RandomCrop(224),
            v_transforms.RandomHorizontalFlip(), 
            v_transforms.Normalize(mean=mean_vals, std=std_vals)
        ])
        train_data_paths = load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent'] == 'German')
        train_dataset = AudioMINST(data_paths=train_data_paths, data_trainsforms=train_tf, include_rate=False)
        args.class_num = 10
        test_tf = Components(transforms=[
            pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((224, 224), antialias=False),
            v_transforms.Normalize(mean=mean_vals, std=std_vals)
        ])
        test_dataset = AudioMINST(data_paths=train_data_paths, data_trainsforms=test_tf, include_rate=False)
        test_dataset = ClipDataset(dataset=test_dataset, rate=args.test_rate)
    else:
        raise Exception('No support')

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    modelF, modelB, modelC = load_models(args)
    store_model_structure_to_txt(model=modelF, output_path=os.path.join(args.full_output_path, 'modelF_structure.txt'))
    store_model_structure_to_txt(model=modelB, output_path=os.path.join(args.full_output_path, 'modelB_structure.txt'))
    store_model_structure_to_txt(model=modelC, output_path=os.path.join(args.full_output_path, 'modelC_structure.txt'))
    optimizer = build_optimizer(args, modelF, modelB, modelC)
    classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth, use_gpu=True).to(device=args.device)

    modelF.train()
    modelB.train()
    modelC.train()

    best_accuracy = 0.
    max_iter = args.max_epoch * len(train_loader)
    iter = 0
    interval = max_iter // args.interval
    ttl_train_loss = 0.
    ttl_train_num = 0
    ttl_train_corr = 0
    for epoch in range(1, args.max_epoch+1):
        print(f'Epoch [{epoch}/{args.max_epoch}]')
        print('Training...')
        for features, labels in tqdm(train_loader):
            features, labels = features.to(args.device), labels.to(args.device)
            outputs = modelC(modelB(modelF(features)))
            loss = classifier_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs.detach(), dim=1)   
            ttl_train_corr += (preds == labels).sum().cpu().item()
            ttl_train_loss += loss.cpu().item()
            ttl_train_num += labels.shape[0]
            features = None
            labels = None
            loss = None
            outputs = None

            if iter % interval == 0 or iter == max_iter-1:
                lr_scheduler(optimizer=optimizer, iter_num=iter, max_iter=max_iter)
            iter += 1

        modelF.eval()
        modelB.eval()
        modelC.eval()
        ttl_corr = 0
        ttl_size = 0
        print('Validating...')
        for features, labels in tqdm(test_loader):
            features, labels = features.to(args.device), labels.to(args.device)
            with torch.no_grad():
                outputs = modelC(modelB(modelF(features)))
                _, preds = torch.max(outputs.detach(), dim=1)
            ttl_corr += (preds == labels).sum().cpu().item()
            ttl_size += labels.shape[0]
        curr_accu = ttl_corr / ttl_size * 100.
        wandb_run.log({'Train/Accuracy': ttl_train_corr/ttl_train_num*100.}, step=epoch)
        wandb_run.log({'Val/Accuracy': curr_accu}, step=epoch)
        wandb_run.log({'Train/classifier_loss': ttl_train_loss/ttl_train_num}, step=epoch)
        wandb_run.log({'Train/learning_rate': optimizer.param_groups[0]['lr']}, step=epoch, commit=True)
        if curr_accu > best_accuracy:
            best_accuracy = curr_accu
            best_modelF = modelF.state_dict()
            best_modelB = modelB.state_dict()
            best_modelC = modelC.state_dict()

            torch.save(best_modelF, os.path.join(args.full_output_path, f'{args.output_weight_prefix}_best_modelF.pt'))
            torch.save(best_modelB, os.path.join(args.full_output_path, f'{args.output_weight_prefix}_best_modelB.pt'))
            torch.save(best_modelC, os.path.join(args.full_output_path, f'{args.output_weight_prefix}_best_modelC.pt'))
        modelF.train()
        modelB.train()
        modelC.train()
        features = None
        labels = None
        outputs = None