import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.toolkit import BatchTransform
from ttt.lib.prepare_dataset import TimeShiftOps

def build_mnist_model(args: argparse.Namespace) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    from ttt.models.RestNet import ResNetMNIST as ResNet
    from ttt.models.SSHead import ExtractorHead

    print('Building model...')
    if args.group_norm == 0:
        norm_layer = nn.BatchNorm2d
    else:
        def gn_helper(planes):
            return nn.GroupNorm(num_groups=args.group_norm, num_channels=planes)
        norm_layer = gn_helper
    net = ResNet(depth=args.depth, width=args.width, channels=1, class_num=args.class_num, 
                 norm_layer=norm_layer, fc_in=args.final_full_line_in)
    if args.shared == 'none':
        args.shared = None

    if args.shared == 'layer3' or args.shared is None:
        from ttt.models.SSHead import extractor_from_layer3
        ext = extractor_from_layer3(net)
        head = nn.Linear(in_features=args.final_full_line_in, out_features=args.ssh_class_num)
    elif args.shared == 'layer2':
        from ttt.models.SSHead import extractor_from_layer2, head_on_layer2
        ext = extractor_from_layer2(net)
        head = head_on_layer2(net=net, width=args.width, class_num=args.ssh_class_num, fc_in=args.final_full_line_in)
    ssh = ExtractorHead(ext=ext, head=head)
    return net.to(device=args.device), ext.to(device=args.device), head.to(device=args.device), ssh.to(device=args.device)

def time_shift_inference(model: nn.Module, loader: DataLoader, test_transf: dict[str, BatchTransform], device: str) -> float:
    test_corr = 0.
    test_size = 0.
    model.eval()
    for inputs, labels in tqdm(loader):
        inputs = test_transf[TimeShiftOps.ORIGIN].transf(inputs)
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)
        test_corr += (preds == labels).sum().cpu().item()
        test_size += labels.shape[0]
    return test_corr / test_size * 100.