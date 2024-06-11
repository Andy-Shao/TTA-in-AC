import argparse
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from lib import models

def test_target(args: argparse.Namespace):
    pass

def load_data(args: argparse.Namespace):
    pass

def op_copy(optimizer: optim.Optimizer):
    pass

def train_source(args: argparse.Namespace):
    dataset_loader = load_data(args)
    # set base network
    if args.model[0:3] == 'res':
        modelF = models.ResBase(res_name=args.net,se=args.se,nl=args.nl).cuda()
    elif args.model[0:3] == 'vgg':
        modelF = models.VGGBase(vgg_name=args.net).cuda()
    elif args.model == 'vit':
        modelF = models.ViT().cuda()

    modelB = models.feat_bootleneck(type=args.classifier, feature_dim=modelF.in_features, bottleneck_dim=args.bottleneck).cuda()
    modelC = models.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    gpu_list = [i for i in range(torch.cuda.device_count())]
    print(f"Let's use {len(gpu_list)} GPUs")
    modelF = nn.DataParallel(modelF, device_ids=gpu_list)
    modelB = nn.DataParallel(modelB, device_ids=gpu_list)
    modelC = nn.DataParallel(modelC, device_ids=gpu_list)

    param_group = []
    learning_rate = args.lr
    for k, v in modelF.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate * .1}]
    for k, v in modelB.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate}]
    for k,v in modelC.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer=optimizer)

def print_args(args: argparse.Namespace):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += f'{arg}:{content}\n'
    return s

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'GPU must be capable'

    ap = argparse.ArgumentParser(description='SHOT')
    ap.add_argument('--source', type=int, default=0, help='source')
    ap.add_argument('--target', type=int, default=0, help='target')
    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--interval', type=int, default=50, help='interval')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--dataset', type=str, default='office-home', choices=['office-home', 'office', 'domain_net'])
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--model', type=str, default='deit_s', help='vgg16, resnet50, resnet101, vit, deit_s')
    ap.add_argument('--seed', type=int, default=2020, help='random seed')
    ap.add_argument('--bottleneck', type=int, default=256)
    ap.add_argument('--epsilon', type=float, default=1e-5)
    ap.add_argument('--layer', type=str, default='wn', choices=['linear', 'wn'])
    ap.add_argument('--classifier', type=str, default='bn', choices=['ori', 'bn'])
    ap.add_argument('--smooth', type=float, default=.1)
    ap.add_argument('--output', type=str, default='pre_train')
    ap.add_argument('--source_path', type=str, default='./data')
    ap.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    ap.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    ap.add_argument('--bsp', type=bool, default=False)
    ap.add_argument('--se', type=bool, default=False)
    ap.add_argument('--nl', type=bool, default=False)
    ap.add_argument('--worker', type=int, default=16)
    ap.add_argument('--wandb', type=int, default=0)

    args = ap.parse_args()

    if args.dataset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dataset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31

    gpu_id = ''
    for i in range(torch.cuda.device_count()):
        gpu_id += str(i) + ','
    gpu_id.removesuffix(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    folder = args.source_path
    args.training_dataset_path = folder + args.dataset + '/' + names[args.source] + '.txt'
    args.test_dataset_path = folder + args.dataset + '/' + names[args.target] + '.txt'

    wandb.init(
        project='ConNMix ECCV', name=f'SRC {names[args.source]}', mode='online' if args.wandb else 'disabled',
        config=args, tags=['SRC', args.dataset, args.model]
    )

    print(print_args(args))
    if args.dataset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.source_classes = [i for i in range(65)]
            args.target_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 25
            args.source_classes = [i for i in range(25)]
            args.target_classes = [i for i in range(64)]
    
    args.output_dir_src = os.path.join(args.output, args.da, args.dataset, names[args.source][0].upper())
    args.name_src = names[args.source][0].upper()
    if not os.path.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not os.path.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)
    
    args.out_file = open(os.path.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args=args) + '\n')
    args.out_file.flush()

    train_source(args=args)

    args.out_file = open(os.path.join(args.output_dir_src, 'log_test.txt'), 'w')
    for i in range(len(names)):
        if i == args.source:
            continue
        args.target = i
        args.name = names[args.source][0].upper() + names[args.target][0].upper()

        folder = args.source_path
        args.source_dataset_path = folder + args.dataset + '/' + names[args.source] + '.txt'
        args.test_dataset_path = folder + args.dataset + '/' + names[args.target] + '.txt'
        if args.dataset == 'domain_net':
            args.txt_eval_dn = folder + args.dataset + '/' + names[args.target] + '_test.txt'
        else:
            args.txt_eval_dn = args.test_dataset_path

        if args.dataset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.source_classes = [i for i in range(65)]
                args.target_classes = [i for i in range(25)]
            if args.da == 'oda':
                args.class_num = 25
                args.source_classes = [i for i in range(25)]
                args.target_classes = [i for i in range(65)]
        test_target(args=args)