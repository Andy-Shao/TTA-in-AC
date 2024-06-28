import argparse
import os
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn

from lib.toolkit import print_argparse
from ttt.lib.test_helpers import build_mnist_model
from ttt.lib.prepare_dataset import prepare_train_data, train_transforms, TimeShiftOps
from ttt.lib.time_shift_rotation import rotate_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
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
    parser.add_argument('--severity_level', default=.0025, type=float)
    parser.add_argument('--shift_limit', default=.25, type=float)

    args = parser.parse_args()
    print('TTT pre-train')
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    args.output_full_path = os.path.join(args.output_path, args.dataset, 'ttt', 'pre_time_shift_train')
    try:
        os.makedirs(args.output_full_path)
    except:
        pass

    if args.dataset == 'audio-mnist':
        args.class_num = 10
        args.sample_rate = 48000
        args.n_mels = 64
        args.final_full_line_in = 384
        args.hop_length = 505
        args.ssh_class_num = 3
    else:
        raise Exception('No support')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.severity = 'high' if args.severity_level >= .01 else 'low'
    print_argparse(args=args)
    # Finished args prepare

    net, ext, head, ssh = build_mnist_model(args=args)
    train_dataset, train_loader = prepare_train_data(args=args)
    tran_transfs = train_transforms(args=args)

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

    accu_records = pd.DataFrame(columns=['dataset', 'type', 'step', 'accuracy', 'loss'])
    train_step = 0
    print('Running...')
    print('Accuracy (%)\t\ttrain\t\tself-supervised')
    for epoch in range(1, args.max_epoch+1):
        net.train()
        ssh.train()

        ttl_cls_corr = 0.
        ttl_ssh_corr = 0.
        ttl_cls_size = 0
        ttl_ssh_size = 0
        # for batch_idx, (features, labels) in enumerate(train_loader):
        for features, labels in tqdm(train_loader):
            optimizer.zero_grad()
            features_cls = tran_transfs[TimeShiftOps.ORIGIN].transf(features).to(args.device)
            labels_cls = labels.to(args.device)
            outputs_cls = net(features_cls)
            loss = criterion(outputs_cls, labels_cls)
            _, preds_cls = torch.max(outputs_cls, dim=1)
            cls_corr = (preds_cls == labels_cls).sum().cpu().item()
            cls_accu = cls_corr / labels_cls.size(0) * 100.
            accu_records.loc[len(accu_records)] = [args.dataset, 'cls', train_step, cls_accu, loss.cpu().item()]
            ttl_cls_corr += cls_corr
            ttl_cls_size += labels_cls.size(0)

            if args.shared is not None:
                features_ssh, labels_ssh = rotate_batch(features, args.rotation_type, data_transforms=tran_transfs)
                features_ssh, labels_ssh = features_ssh.to(args.device), labels_ssh.to(args.device)
                outputs_ssh = ssh(features_ssh)
                loss_ssh = criterion(outputs_ssh, labels_ssh)
                loss += loss_ssh
                _, preds_ssh = torch.max(outputs_ssh, dim=1)
                ssh_corr = (preds_ssh == labels_ssh).sum().cpu().item()
                ssh_accu = ssh_corr / labels_ssh.size(0) * 100.
                accu_records.loc[len(accu_records)] = [args.dataset, 'ssh', train_step, ssh_accu, loss_ssh.cpu().item()]
                ttl_ssh_corr += ssh_corr
                ttl_ssh_size += labels_ssh.size(0)
            loss.backward()
            optimizer.step()
            train_step += 1
        scheduler.step()
        print(('Epoch %d/%d:' %(epoch, args.max_epoch)).ljust(24) + 'cls_accu: %.2f\t\t ssh_accu: %.2f' %(ttl_cls_corr/ttl_cls_size*100., ttl_ssh_corr/ttl_ssh_size*100.))
    accu_records.to_csv(os.path.join(args.output_full_path, 'accu_record.csv'))
    state = {'net': net.state_dict(), 'head': head.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, os.path.join(args.output_full_path, f'ckpt_{args.severity}.pth'))