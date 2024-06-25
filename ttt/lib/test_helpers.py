import argparse

import torch.nn as nn

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
                 norm_layer=norm_layer, fc_in=args.final_full_line_in).to(device=args.device)
    if args.shared == 'none':
        args.shared = None

    if args.shared == 'layer3' or args.shared is None:
        from ttt.models.SSHead import extractor_from_layer3
        ext = extractor_from_layer3(net)
        head = nn.Linear(in_features=64 * args.width, out_features=4)
    elif args.shared == 'layer2':
        from ttt.models.SSHead import extractor_from_layer2, head_on_layer2
        ext = extractor_from_layer2(net)
        head = head_on_layer2(net=net, width=args.width, class_num=3)
    ssh = ExtractorHead(ext=ext, head=head).to(device=args.device)
    return net, ext, head, ssh