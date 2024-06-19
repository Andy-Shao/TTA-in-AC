import argparse
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

NORM_MEAN, NORM_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
test_transforms = transforms.Compose(transforms=[
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])
train_transforms = transforms.Compose(transforms=[
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])
common_corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
	'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
	'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

def prepare_test_data(args: argparse.Namespace) -> tuple[Dataset, DataLoader]:
    if args.dataset == 'cifar10':
        test_size = 10000
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            print('Test on the original test set')
            test_set = torchvision.datasets.CIFAR10(root=args.dataroot, train=False, download=True, transform=test_transforms)
        elif args.corrupion in common_corruptions:
            print('Test on %s level %d' %(args.corruption, args.level))
            test_set_raw = np.load(file=f'{args.dataroot}/CIFAR-10-C/{args.corruption}.npy')
            test_set_raw = test_set_raw[(args.level-1)*test_size: (args.level-1)*test_size]
            test_set = torchvision.datasets.CIFAR10(root=args.dataroot, train=False, download=True, transform=test_transforms)
            test_set.data = test_set_raw
        elif args.corruption == 'cifar_new':
            from lib.cifar_new import CIFAR_New
            print('Test on CIFAR-10.1')
            test = CIFAR_New(root=args.dataroot + 'CIFAR-10.1/datasets/', transform=test_transforms)
            permute = False
        else:
            raise Exception('Corruption not found!')
    else:
        raise Exception('Dataset not found!')
    
    if not hasattr(args, 'workers'):
        args.workers = 1
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return test_set, test_loader

def prepare_train_data(args: argparse.Namespace) -> tuple[Dataset, DataLoader]:
    print('Preparing data...')
    if args.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=args.dataroot, train=True, download=True, transform=train_transforms)
    else:
        raise Exception('Dataset not found!')
    
    if not hasattr(args, 'workers'):
        args.workers = 1
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    return train_set, train_loader