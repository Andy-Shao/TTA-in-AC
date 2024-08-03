import argparse
import os
import numpy as np
import random
import wandb
from tqdm import tqdm

import torch 
from torchaudio import transforms as a_transforms
from torchvision import transforms as v_transforms
from torch.utils.data import DataLoader
from torch import optim

from lib.toolkit import print_argparse, cal_norm, store_model_structure_to_txt
from lib.wavUtils import pad_trunc, time_shift, Components
from CoNMix.lib.prepare_dataset import ExpandChannel
from lib.scDataset import SpeechCommandsDataset
from CoNMix.pre_train import load_models, build_optimizer
from CoNMix.lib.loss import CrossEntropyLabelSmooth
from lib.datasets import ClipDataset

def lr_scheduler(optimizer: torch.optim.Optimizer, iter_num: int, max_iter: int, step:int, gamma=10, power=0.75) -> optim.Optimizer:
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = .9
        param_group['nestenv'] = True
    wandb.log({'Train/learning_rate': param_group['lr']}, step=step)
    return optimizer

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands'])
    ap.add_argument('--dataset_root_path', type=str)
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
    args.class_num = 30
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
    
    max_ms = 1000
    sample_rate = 16000
    n_mels=128
    hop_length=125
    tf_array = [
        pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
        time_shift(shift_limit=.25, is_random=True, is_bidirection=True),
        a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
        a_transforms.AmplitudeToDB(top_db=80),
        # a_transforms.FrequencyMasking(freq_mask_param=.1),
        # a_transforms.TimeMasking(time_mask_param=.1),
        ExpandChannel(out_channel=3),
        v_transforms.Resize((256, 256), antialias=False),
        v_transforms.RandomCrop(224),
        v_transforms.RandomHorizontalFlip(), 
    ]
    if args.normalized:
        print('calculate the train dataset mean and standard deviation')
        train_tf = Components(transforms=tf_array)
        train_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='train', include_rate=False, data_tfs=train_tf)
        train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=False, drop_last=False)
        train_mean, train_std = cal_norm(loader=train_loader)
        tf_array.append(v_transforms.Normalize(mean=train_mean, std=train_std))
    train_tf = Components(transforms=tf_array)
    train_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='train', include_rate=False, data_tfs=train_tf)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    tf_array = [
        pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
        a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
        a_transforms.AmplitudeToDB(top_db=80),
        ExpandChannel(out_channel=3),
        v_transforms.Resize((224, 224), antialias=False),
    ]
    if args.normalized:
        print('calculate the validation dataset mean and standard deviation')
        val_tf = Components(transforms=tf_array)
        val_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='train', include_rate=False, data_tfs=val_tf)
        val_loader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=False, drop_last=False)
        val_mean, val_std = cal_norm(loader=val_loader)
        tf_array.append(v_transforms.Normalize(mean=val_mean, std=val_std))
    val_tf = Components(transforms=tf_array)
    val_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='train', include_rate=False, data_tfs=val_tf)
    val_dataset = ClipDataset(dataset=val_dataset, rate=args.test_rate)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    modelF, modelB, modelC = load_models(args)
    store_model_structure_to_txt(model=modelF, output_path=os.path.join(args.full_output_path, 'modelF_structure.txt'))
    store_model_structure_to_txt(model=modelB, output_path=os.path.join(args.full_output_path, 'modelB_structure.txt'))
    store_model_structure_to_txt(model=modelC, output_path=os.path.join(args.full_output_path, 'modelC_structure.txt'))
    optimizer = build_optimizer(args, modelF=modelF, modelB=modelB, modelC=modelC)
    classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth, use_gpu=True).to(device=args.device)

    modelF.train()
    modelB.train()
    modelC.train()

    best_accuracy = 0.
    max_iter = args.max_epoch * len(train_loader)
    iter = 0
    interval = max_iter // args.interval
    for epoch in range(1, args.max_epoch+1):
        print(f'Epoch [{epoch}/{args.max_epoch}]')
        ttl_train_loss = 0.
        ttl_train_num = 0
        for features, labels in tqdm(train_loader):
            features, labels = features.to(args.device), labels.to(args.device)
            outputs = modelC(modelB(modelF(features)))
            loss = classifier_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ttl_train_loss += loss.cpu().item()
            ttl_train_num += labels.shape[0]
            features = None
            labels = None
            loss = None
            outputs = None

            if iter % interval == 0 or iter == max_iter-1:
                lr_scheduler(optimizer=optimizer, iter_num=iter, max_iter=max_iter, step=iter//interval)

                modelF.eval()
                modelB.eval()
                modelC.eval()
                ttl_corr = 0
                ttl_size = 0
                for features, labels in val_loader:
                    features, labels = features.to(args.device), labels.to(args.device)
                    with torch.no_grad():
                        outputs = modelC(modelB(modelF(features)))
                        _, preds = torch.max(outputs, dim=1)
                    ttl_corr += (preds == labels).sum().cpu().item()
                    ttl_size += labels.shape[0]
                curr_accu = ttl_corr / ttl_size * 100.
                wandb_run.log({'Train/Accuracy': curr_accu}, step=iter//interval)
                wandb_run.log({'Train/classifier_loss': ttl_train_loss/ttl_train_num}, step=iter//interval)
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
            iter += 1