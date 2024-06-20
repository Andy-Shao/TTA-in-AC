import argparse
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from lib.misc import my_makedir
from lib.test_helpers import build_model
from lib.prepare_dataset import prepare_test_data

def test_single(model: nn.Module, image: Image, label: torch.Tensor) -> tuple[int, torch.Tensor]:
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument('--corruption', default='original')
    parser.add_argument('--dataroot', default='/data/yusun/datasets/')
    parser.add_argument('--shared', default=None)
    ########################################################################
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--group_norm', default=0, type=int)
    parser.add_argument('--fix_bn', action='store_true')
    parser.add_argument('--fix_ssh', action='store_true')
    ########################################################################
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--niter', default=1, type=int)
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--threshold', default=1, type=float)
    parser.add_argument('--dset_size', default=0, type=int)
    ########################################################################
    parser.add_argument('--outf', default='.')
    parser.add_argument('--resume', default=None)

    args = parser.parse_args()
    args.threshold += 0.001		# to correct for numeric errors
    my_makedir(args.outf)
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    net, ext, head, ssh = build_model(args=args)
    test_set, test_loader = prepare_test_data(args=args)
    printed_head = 'test_adapt:'

    print('%s Resuming from %s...' %(printed_head, args.resume))
    ckpt = torch.load(args.resume + '/ckpt.pth')
    if args.online:
        net.load_state_dict(ckpt['net'])
        head.load_state_dict(ckpt['head'])
    
    criterion_ssh = nn.CrossEntropyLoss().cuda()
    if args.fix_ssh:
        optimizer_ssh = optim.SGD(params=ext.parameters(), lr=args.lr)
    else:
        optimizer_ssh = optim.SGD(params=ssh.parameters(), lr=args.lr)

    print(f'{printed_head} Running...')
    correct = []
    ssh_conf = []
    train_error = []
    if args.dset_size == 0:
        args.dset_size = len(test_set)
    
    for i in tqdm(range(1, args.dset_size+1)):
        if not args.online:
            net.load_state_dict(ckpt['net'])
            head.load_state_dict(ckpt['head'])

        _, label = test_set[i - 1]
        image = Image.fromarray(test_set.data[i - 1])
