import argparse
import os
import random
import numpy as np
import wandb
from tqdm import tqdm

import torch 
from torch import optim
from torch import nn

from lib.toolkit import print_argparse, count_ttl_params
from ttt.lib.test_helpers import build_sc_model
from ttt.lib.speech_commands.prepare_dataset import prepare_data, train_transforms, val_transforms
from ttt.lib.prepare_dataset import TimeShiftOps
from ttt.lib.time_shift_rotation import rotate_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands'])
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--max_epoch', type=int, default=75)
    parser.add_argument('--lr', type=float, default=.1)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--dataset_root_path', type=str)
    parser.add_argument('--output_path', type=str, default='./result')
    parser.add_argument('--shared', type=str, default='layer2')
    parser.add_argument('--milestone_1', default=50, type=int)
    parser.add_argument('--milestone_2', default=65, type=int)
    parser.add_argument('--rotation_type', default='rand')
    parser.add_argument('--group_norm', default=0, type=int)
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--shift_limit', default=.25, type=float)
    parser.add_argument('--output_csv_name', type=str, default='accu_record.csv')
    parser.add_argument('--output_weight_name', type=str, default='ckpt.pth')
    parser.add_argument('--seed', default=2024, type=int)

    args = parser.parse_args()
    print('TTT pre-train')
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    args.output_full_path = os.path.join(args.output_path, args.dataset, 'ttt', 'pre_time_shift_train')
    try:
        os.makedirs(args.output_full_path)
    except:
        pass

    if args.dataset == 'speech-commands':
        args.class_num = 30
        args.sample_rate = 16000
        args.n_mels = 64
        args.final_full_line_in = 256
        args.hop_length = 253
        args.ssh_class_num = 3
    else:
        raise Exception('No support')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print_argparse(args=args)
    # Finished args prepare
    #############################################################

    net, ext, head, ssh = build_sc_model(args=args)
    print(f'net weight number is: {count_ttl_params(net)}, ssh weight number is: {count_ttl_params(ssh)}, ext weight number is: {count_ttl_params(ext)}')
    print((f'total weight number is: {count_ttl_params(net) + count_ttl_params(head)}'))
    train_dataset, train_loader = prepare_data(args=args, mode='train')
    tran_transfs = train_transforms(args=args)
    val_dataset, val_loader = prepare_data(args=args, mode='validation')
    val_transfs = val_transforms(args=args)

    parameters = list(net.parameters()) + list(head.parameters())
    optimizer = optim.SGD(params=parameters, lr=args.lr, momentum=.9, weight_decay=5e-4)
    """
    # Assuming optimizer uses lr = 0.05 for all groups
    # lr = 0.05     if epoch < 30
    # lr = 0.005    if 30 <= epoch < 80
    # lr = 0.0005   if epoch >= 80
    scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    """
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[args.milestone_1, args.milestone_2], gamma=.1, last_epoch=-1)
    criterion = nn.CrossEntropyLoss().to(device=args.device)

    wandb_run = wandb.init(
        project=f'Audio Classification Pre-Training ({args.dataset})', name=f'TTT', mode='online' if args.wandb else 'disabled',
        config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])
    
    max_val_accu = 0.
    for epoch in range(args.max_epoch):
        net.train()
        ssh.train()
        ttl_corr = 0
        ttl_size = 0
        ttl_loss = 0
        ttl_ssh_corr = 0
        ttl_ssh_size = 0
        print(f'{epoch+1}/{args.max_epoch} training...')
        for features, labels in tqdm(train_loader):
            optimizer.zero_grad()
            features_cls = tran_transfs[TimeShiftOps.ORIGIN].transf(features).to(args.device)
            labels_cls = labels.to(args.device)
            outputs_cls = net(features_cls)
            loss = criterion(outputs_cls, labels_cls)
            _, preds_cls = torch.max(outputs_cls.detach(), dim=1)
            ttl_corr += (preds_cls == labels_cls).sum().cpu().item()
            ttl_size += labels_cls.size(0)
            ttl_loss += loss.cpu().item()

            if args.shared is not None:
                features_ssh, labels_ssh = rotate_batch(features, args.rotation_type, data_transforms=tran_transfs)
                features_ssh, labels_ssh = features_ssh.to(args.device), labels_ssh.to(args.device)
                outputs_ssh = ssh(features_ssh)
                loss_ssh = criterion(outputs_ssh, labels_ssh)
                loss += loss_ssh
                _, preds_ssh = torch.max(outputs_ssh.detach(), dim=1)
                ttl_ssh_corr += (preds_ssh == labels_ssh).sum().cpu().item()
                ttl_ssh_size += labels_ssh.size(0)
            loss.backward()
            optimizer.step()
        scheduler.step()
        wandb.log({'Train/Accuracy':ttl_corr/ttl_size*100., 'Train/classifier_loss':ttl_loss/ttl_size}, step=epoch)
        net.eval()
        ssh.eval()
        ttl_corr = 0
        ttl_size = 0
        print(f'{epoch+1}/{args.max_epoch} validating...')
        for features, labels in tqdm(val_loader):
            features_cls = val_transfs[TimeShiftOps.ORIGIN].transf(features).to(args.device)
            labels_cls = labels.to(args.device)
            with torch.no_grad():
                outputs_cls = net(features_cls)
            _, preds_cls = torch.max(outputs_cls.detach(), dim=1)
            ttl_corr += (preds_cls == labels_cls).sum().cpu().item()
            ttl_size += labels.shape[0]
        val_accu = ttl_corr/ttl_size*100.
        wandb.log({'Val/Accuracy':val_accu}, step=epoch)
        if max_val_accu < val_accu:
            max_val_accu = val_accu
            state = {'net': net.state_dict(), 'head': head.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, os.path.join(args.output_full_path, args.output_weight_name))