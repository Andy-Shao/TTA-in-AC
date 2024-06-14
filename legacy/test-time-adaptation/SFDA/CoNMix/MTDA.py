import argparse
import os
from typing import Tuple, Dict
import csv
import wandb
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from helper.data_list import ImageList_MixUp
from lib import models
from helper.mixup_utils import progress_bar

def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lambda_: float) -> torch.Tensor:
    return lambda_ * criterion(pred, y_a) + (1 - lambda_) * criterion(pred, y_b)

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha=1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lambda_ = np.random.beta(alpha, alpha)
    else:
        lambda_ = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lambda_ * x + (1 - lambda_) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lambda_

def image_train(resize_size=256, crop_size=224, alexnet=False) -> nn.Module:
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(transforms=[
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def test(epoch: int, test_loader: DataLoader, models: Tuple[nn.Module, nn.Module, nn.Module], criterion: nn.Module) -> Tuple[float, float]:
    global best_accuracy
    modelF, modelB, modelC = models
    modelF.eval()
    modelB.eval()
    modelC.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_len = len(test_loader)
    with torch.no_grad():
        for batch_idx, (inputs, pseudo_label, targets, domain) in enumerate(test_loader):
            inputs, pseudo_label, targets, domain = inputs.cuda(), pseudo_label.cuda(), targets.cuda(), domain.cuda()
            inputs, pseudo_label = Variable(inputs), Variable(pseudo_label)
            outputs = modelC(modelB(modelF(inputs)))
            loss = criterion(outputs, pseudo_label)

            test_loss += loss.item()
            _, predicated = torch.max(outputs.data, dim=1)
            total += targets.size(0)
            correct += predicated.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    accuracy = 100. * correct / total
    if accuracy > best_accuracy:
        best_accuracy = accuracy
    return test_loss/test_len, 100. * correct / total

def lr_scheduler(optimizer: optim.Optimizer, iter_num: int, max_iter: int, gamma=10, power=0.75) -> optim.Optimizer:
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
        # wandb.log({'MISC/LR': param_group['lr']})
    return optimizer

def checkpoint(args, modelF, modelB, modelC):
    # Save checkpoint.
    save_path = os.path.join(args.save_weights, args.dataset, names[args.source])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(modelF.state_dict(), os.path.join(save_path, "target_F.pt"))
    torch.save(modelB.state_dict(), os.path.join(save_path, "target_B.pt"))
    torch.save(modelC.state_dict(), os.path.join(save_path, "target_C.pt"))
    print('Model saved to ', save_path )

def train(args: argparse.Namespace, epoch: int, all_loader: DataLoader, optimizer: optim.Optimizer, models: Tuple[nn.Module, nn.Module, nn.Module], criterion: nn.Module) -> Tuple[float, float, float]:
    print('\nEpoch: %d' % epoch)
    modelF, modelB, modelC = models
    modelF.train()
    modelB.train()
    modelC.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    batch_len = len(all_loader)

    for batch_idx, (inputs, pseudo_label, targets, domain) in enumerate(all_loader):
        inputs, pseudo_label, targets, domain = inputs.cuda(), pseudo_label.cuda(), targets.cuda(), domain.cuda()
        inputs, targets_a, targets_b, lambda_ = mixup_data(inputs, pseudo_label, args.alpha)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        # inputs.requires_grad = True
        # targets_a.requires_grad = True
        # targets_b.requires_grad = True

        outputs = modelC(modelB(modelF(inputs)))
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lambda_)
        train_loss += loss.item()
        _, predicated = torch.max(outputs.data, dim=1)
        total += pseudo_label.size(0)
        correct += (lambda_ * predicated.eq(targets_a.data).cpu().sum().float() + (1 - lambda_) * predicated.eq(targets_b.data).cpu().sum().float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({'Train/train_loss': train_loss/(batch_idx+1), 'Train/train_acc': 100.*correct/total})

        progress_bar(batch_idx, len(all_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1),100.*correct/total, correct, total))

    return train_loss/batch_len, reg_loss/batch_len, 100. * correct / total

def op_copy(optimizer: optim.Optimizer) -> optim.Optimizer:
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def init_src_model_load(args: argparse.Namespace) -> Tuple[nn.Module, nn.Module, nn.Module]:
    ## set base network
    if args.net[0:3] == 'res':
        modelF = models.ResBase(res_name=args.net, se=args.se, nl=args.nl).cuda()
    elif args.net[0:3] == 'vgg':
        modelF = models.VGGBase(vgg_name=args.net).cuda()
    elif args.net == 'vit':
        modelF = models.ViT().cuda()
    elif args.net == 'deit_s':
        modelF = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).cuda()
        modelF.in_features = 1000
    
    modelB = models.feat_bootleneck(type='bn', feature_dim=modelF.in_features, bottleneck_dim=256).cuda()
    modelC = models.feat_classifier(type='wn', class_num=args.class_num, bottleneck_dim=256).cuda()

    return modelF, modelB, modelC

def load_data(args: argparse.Namespace) -> Tuple[Dict, Dict]:
    datasets = {}
    dataset_loaders = {}
    
    txt_target = open(f'{args.txt_folder}/{args.dataset}/{names[args.source]}.csv').readlines()
    print("Source Domain: ", names[args.source], "No. of Images: ", len(txt_target))
    datasets['train'] = ImageList_MixUp(txt_target, transform=image_train())
    dataset_loaders['train'] = DataLoader(dataset=datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)

    datasets['test'] = datasets['train']
    dataset_loaders['test'] = dataset_loaders['train']
    return dataset_loaders, datasets

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
    parser.add_argument('--source', default=0, type=int)
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
    all_loader, all_dataset = load_data(args)   
    modelF, modelB, modelC = init_src_model_load(args)
    modelF, modelB, modelC = modelF.cuda(), modelB.cuda(), modelC.cuda()
    cudnn.benchmark = True
    print('Using CUDA..')

    criterion = nn.CrossEntropyLoss()
    param_group = []
    for k,v in modelF.named_parameters():
        param_group += [{'params':v, 'lr':args.lr * .1}]
    for k,v in modelB.named_parameters():
        param_group += [{'params':v, 'lr':args.lr}]
    for k,v in modelC.named_parameters():
        param_group += [{'params':v, 'lr':args.lr}]

    optimizer = optim.SGD(param_group, lr=args.lr, momentum=0.9, weight_decay=args.decay)
    optimizer = op_copy(optimizer)

    log_name = ('results/log' + '.csv')
    if not os.path.exists(log_name):
        with open(log_name, 'w') as log_file:
            log_writer = csv.writer(log_file, delimiter=',')
            log_writer.writerow(['epoch', 'train loss', 'reg loss', 'train acc', 'test loss', 'test acc'])
    
    mode = 'online' if args.wandb else 'disabled'
    wandb.init(project='CoNMix ECCV MTDA', entity='vclab', name=f'MTDA {names[args.source]} to Others '+ args.suffix, reinit=True, mode=mode, config=args, tags=[args.dataset, args.net, 'MTDA'])

    print(f'\nStarting training {names[args.source]} to others.')
    train_len = len(all_dataset['train'])
    test_len = len(all_dataset['test'])
    print(f'Training: {train_len} Images \t Testing: {test_len} Images')

    for epoch in range(start_epoch, args.epoch+1):
        train_loss, reg_loss, train_accuracy = train(args, epoch, all_loader['train'], optimizer, models=(modelF, modelB, modelC), criterion=criterion)
        checkpoint(args, modelF, modelB, modelC)
        optimizer = lr_scheduler(optimizer, iter_num=epoch, max_iter=args.epoch)
        if epoch % args.interval == 0:
            print('\n Start Testing')
            test_loss, test_accuracy = test(epoch,all_loader['test'], models=(modelF, modelB, modelC), criterion=criterion)
            wandb.log({ 'Test/test_loss': test_loss,  'Test/test_acc': test_accuracy})