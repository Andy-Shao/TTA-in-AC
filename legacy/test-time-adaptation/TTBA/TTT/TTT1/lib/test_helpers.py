import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.rotation import rotate_batch
from lib.misc import flat_grad
from lib.rotation import rotate_batch

def build_model(args: argparse.Namespace) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    from models.RestNet import ResNetCifar as ResNet
    from models.SSHead import ExtractorHead
    print('Building model...')
    if args.dataset[:7] == 'cifar10':
        class_num = 10
    elif args.dataset == 'cifar7':
        if not hasattr(args, 'modified') or args.modified:
            class_num = 7
        else:
            class_num = 10
    
    if args.group_norm == 0:
        norm_layer = nn.BatchNorm2d
    else:
        def gn_helper(planes):
            return nn.GroupNorm(num_groups=args.group_norm, num_channels=planes)
        norm_layer = gn_helper
    net = ResNet(depth=args.depth, width=args.width, channels=3, class_num=class_num, norm_layer=norm_layer).cuda()
    if args.shared == 'none':
        args.shared = None
    
    if args.shared == 'layer3' or args.shared is None:
        from models.SSHead import extractor_from_layer3
        ext = extractor_from_layer3(net)
        head = nn.Linear(in_features=64 * args.width, out_features=4)
    elif args.shared == 'layer2':
        from models.SSHead import extractor_from_layer2, head_on_layer2
        ext = extractor_from_layer2(net)
        head = head_on_layer2(net=net, width=args.width, class_num=4)
    ssh = ExtractorHead(ext=ext, head=head).cuda()

    if hasattr(args, 'parallel') and args.parallel:
        net = torch.nn.DataParallel(net)
        ssh = torch.nn.DataParallel(ssh)
    return net, ext, head, ssh

def test(dataloader: DataLoader, model: nn.Module, sslabel=None) -> tuple[float, np.ndarray, np.ndarray]:
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    model.eval()
    correct = []
    losses = []
    for batch_idx, (features, labels) in enumerate(dataloader):
        if sslabel is not None:
            features, labels = rotate_batch(features, sslabel)
        features, labels = features.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(features)
            loss = criterion(outputs, labels)
            losses.append(loss.cpu())
            _, predicted = torch.max(input=outputs, dim=1)
            correct.append(predicted.eq(other=labels).cpu())
    correct = torch.cat(correct).numpy()
    losses = torch.cat(losses).numpy()
    model.train()
    return 1 - correct.mean(), correct, losses

def plot_epochs(all_err_cls: list, all_err_ssh: list, fname: str, use_agg=True) -> None:
    import matplotlib.pyplot as plt
    if use_agg:
        plt.switch_backend('agg')
        
    plt.plot(np.asarray(all_err_cls)*100, color='r', label='classifier')
    plt.plot(np.asarray(all_err_ssh)*100, color='b', label='self-supervised')
    plt.xlabel('epoch')
    plt.ylabel('test error (%)')
    plt.legend()
    plt.savefig(fname)
    plt.close()

def test_grad_corr(dataloader: DataLoader, net: nn.Module, ssh: nn.Module, ext: nn.Module) -> list:
    criterion = nn.CrossEntropyLoss().cuda()
    net.eval()
    ssh.eval()
    corr = []
    for batch_idx, (features, labels) in enumerate(dataloader):
        net.zero_grad()
        ssh.zero_grad()
        features_cls, labels_cls = features.cuda(), labels.cuda()
        outputs_cls = net(features_cls)
        loss_cls = criterion(outputs_cls, labels_cls)
        grad_cls = torch.autograd.grad(outputs=loss_cls, inputs=ext.parameters())
        grad_cls = flat_grad(grad_cls)

        ext.zero_grad()
        features, labels = rotate_batch(batch=features, label='expand')
        features_ssh, labels_ssh = features.cuda(), labels.cuda()
        outputs_ssh = ssh(features_ssh)
        loss_ssh = criterion(outputs_ssh, labels_ssh)
        grad_ssh = torch.autograd.grad(outputs=loss_ssh, inputs=ext.parameters())
        grad_ssh = flat_grad(grad_ssh)

        corr.append(torch.dot(grad_cls, grad_ssh).item())
    net.train()
    ssh.train()
    return corr

def count_each(tuple_: tuple) -> list:
	return [item.sum() for item in tuple_]

def pair_buckets(o1: np.ndarray, o2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	crr = np.logical_and( o1, o2 )
	crw = np.logical_and( o1, np.logical_not(o2) )
	cwr = np.logical_and( np.logical_not(o1), o2 )
	cww = np.logical_and( np.logical_not(o1), np.logical_not(o2) )
	return crr, crw, cwr, cww