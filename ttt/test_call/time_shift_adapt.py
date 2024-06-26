import argparse
import os

import torch 
import torch.nn as nn

from lib.toolkit import print_argparse
from ttt.lib.test_helpers import build_mnist_model
from ttt.lib.prepare_dataset import test_transforms, prepare_test_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--severity_level', default=.0025, type=int)
    parser.add_argument('--corruption', default='original', type=str)
    parser.add_argument('--dataset_root_path', type=str)
    parser.add_argument('--shared', default=None)
    ########################################################################
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--group_norm', default=0, type=int)
    parser.add_argument('--fix_bn', action='store_true')
    parser.add_argument('--fix_ssh', action='store_true')
    ########################################################################
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--niter', default=1, type=int)
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--threshold', default=1., type=float)
    ########################################################################
    parser.add_argument('--output_path', default='./result')
    parser.add_argument('--origin_model_weight_file_path', type=str)

    args = parser.parse_args()
    args.threshold += 0.001		# to correct for numeric errors
    args.output_full_path = os.path.join(args.output_path, args.dataset, 'ttt', 'time_shift_adapt')
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
        args.shift_limit = .25
        args.ssh_class_num = 3
    else:
        raise Exception('No support')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_argparse(args=args)
    # Finished args prepare

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    if args.dataset == 'audio-mnist':
        net, ext, head, ssh = build_mnist_model(args)
    test_transfs = test_transforms(args)
    test_dataset, test_loader = prepare_test_data(args)

    print('test_adapt Resuming from %s...' %(args.origin_model_weight_file_path))
    stat = torch.load(args.origin_model_weight_file_path)
    if args.online:
        net.load_state_dict(stat['net'])
        head.load_state_dict(stat['head'])
    
    criterion_ssh = nn.CrossEntropyLoss().to(args.device)