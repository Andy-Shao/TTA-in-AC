import argparse
import os

import torch
from torchvision import transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    # parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--net', default="deit_s", type=str, help='model type (default: ResNet18)')
    parser.add_argument('--worker', type=int, default=8, help="number of workers")

    parser.add_argument('--kd', type=bool, default=False)
    parser.add_argument('--se', type=bool, default=False)
    parser.add_argument('--nl', type=bool, default=False)
    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--suffix', default='0', type=str, help=' name of run')
    parser.add_argument('--seed', default=2022, type=int, help='random seed')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')

    parser.add_argument('--epoch', default=100, type=int, help='total epochs to run')
    parser.add_argument('--interval', default=2, type=int)
    parser.add_argument('--no-augment', dest='augment', action='store_false', help='use standard augmentation (default: True)')
    parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--alpha', default=1., type=float, help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--s', default=0, type=int)
    parser.add_argument('--txt_folder', default='csv', type=str)
    parser.add_argument('--save_weights', default='MTDA_weights', type=str)
    parser.add_argument('--dataset', type=str, default='office-home', choices=['visda-2017', 'office', 'office-home', 'office-caltech', 'pacs', 'domain_net'])
    parser.add_argument('--wandb', type=int, default=1)

    args = parser.parse_args()
    gpu_id = ''
    for i in range(torch.cuda.device_count()):
        gpu_id += str(i) + ','
    gpu_id.removesuffix(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    args.batch_size = args.batch_size * torch.cuda.device_count()
    use_cuda = torch.cuda.is_available()

    best_accuracy = 0
    start_epoch = 1

    if args.seed != 0:
        torch.manual_seed(args.seed)
    
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset =='domain_net':
        names = ['clipart', 'infograph', 'painting', 'quickdraw','sketch', 'real']
        args.class_num = 345

    # Data
    print('==> Preparing data..')
    if args.augment:
        transform_train = transforms.Compose(transforms=[
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose(transforms=[
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])

    if not os.path.isdir('results'):
        os.mkdir('results')

    print('==> Perparing Dataloaders and Building model..')