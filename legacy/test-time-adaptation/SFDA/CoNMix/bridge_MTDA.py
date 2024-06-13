import argparse
import os
import random
import numpy as np
from typing import Tuple, Dict
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from torchvision import transforms

from lib import models
from helper.data_list import ImageList_idx
from lib.loss import Entropy

def image_test(resize_size=256, crop_size=224, alexnet=False) -> nn.Module:
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        assert False, 'no support'
    return transforms.Compose(transforms=[
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def image_train(resize_size=256, crop_size=224, alexnet=False) -> nn.Module:
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        assert False, 'no support'
    return transforms.Compose(transforms=[
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def cal_acc(loader: DataLoader, modelF: nn.Module, modelB: nn.Module, modelC: nn.Module, flag=False) -> Tuple:
    start_test = True
    print("Finding Accuracy and Pseudo Label")
    with torch.no_grad():
        iter_test = iter(loader)
        all_idx = []
        for i in tqdm(range(len(loader))):
            inputs, labels, idx = next(iter_test)
            inputs = inputs.cuda()
            features = modelB(modelF(inputs))
            outputs = modelC(features)
            if start_test:
                all_feature = features.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
                all_idx = idx.int()
            else:
                all_feature = torch.cat((all_feature, features.float().cpu()), dim=0)
                all_output = torch.cat((all_output, outputs.float().cpu()), dim=0)
                all_label = torch.cat((all_label, labels.float()), dim=0)
                all_idx = torch.cat((all_idx, idx.int()), dim=0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, dim=1)

    all_feature = torch.cat((all_feature, torch.ones(all_feature.size(0), 1)), dim=1)
    all_feature = (all_feature.t()/torch.norm(all_feature, p=2, dim=1)).t()
    all_feature = all_feature.float().cpu().numpy()

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent, predict, all_idx.numpy()

def cal_acc_oda(loader: DataLoader, modelF: nn.Module, modelB: nn.Module, modelC: nn.Module) -> Tuple:
    pass

def load_data(args: argparse.Namespace) -> Dict:
    datasets = {}
    dataset_loaders = {}
    train_batch_size = args.batch_size
    txt_source = open(args.source_dataset_path).readlines()
    txt_test = open(args.test_dataset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.source_classes)):
            label_map_s[args.source_classes[i]] = i
    
        new_source = []
        for i in range(len(txt_source)):
            rec = txt_source[i].strip().split(' ')
            feature_path, label = rec[0], rec[1]
            if int(label) in args.source_classes:
                line = feature_path + ' ' + str(label_map_s[int(label)]) + '\n'
                new_source.append(line)
        txt_source = new_source.copy()

        new_target = []
        for i in range(len(txt_test)):
            rec = txt_test[i].strip().split(' ')
            feature_path, label = rec[0], rec[1]
            if int(label) in args.target_classes:
                if int(label) in args.source_classes:
                    line = feature_path + ' ' + str(label_map_s[int(label)]) + '\n'
                    new_target.append(line)
                else:
                    line = feature_path + ' ' + str(len(label_map_s)) + '\n'
                    new_target.append(line)
        txt_test = new_target.copy()

    if args.trte == 'val':
        datasize = len(txt_source)
        target_size = int(.9 * datasize)
        target_txt, test_txt = torch.utils.data.random_split(txt_source, [target_size, datasize - target_size])
    else:
        datasize = len(txt_source)
        target_size = int(.9 * datasize)
        _, test_txt = torch.utils.data.random_split(txt_source, [target_size, datasize - target_size])
        target_txt = txt_source

    datasets['source_tr'] = ImageList_idx(target_txt, transform=image_train())
    dataset_loaders["source_tr"] = DataLoader(datasets["source_tr"], batch_size=train_batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
    datasets["source_te"] = ImageList_idx(test_txt, transform=image_test())
    dataset_loaders["source_te"] = DataLoader(datasets["source_te"], batch_size=train_batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
    datasets["test"] = ImageList_idx(txt_test, transform=image_test())
    dataset_loaders["test"] = DataLoader(datasets["test"], batch_size=train_batch_size*2, shuffle=True, num_workers=args.worker, drop_last=False)
        
    return dataset_loaders

def test_target(args: argparse.Namespace) -> Tuple:
    dataset_loaders = load_data(args)
    ## set base network
    if args.net[0:3] == 'res':
        modelF = models.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        modelF = models.VGGBase(vgg_name=args.net).cuda()  
    elif args.net[0:4] == 'deit':
        if args.net == 'deit_s':
            modelF = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).cuda()
        elif args.net == 'deit_b':
            modelF = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).cuda()
        modelF.in_features = 1000
    else:
        modelF = models.ViT().cuda()

    modelB = models.feat_bootleneck(type=args.classifier, feature_dim=modelF.in_features, bottleneck_dim=args.bottleneck).cuda()
    modelC = models.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    gpu_list = [i for i in range(torch.cuda.device_count())]
    print(f"Let's use {len(gpu_list)} GPUs")
    modelF = nn.DataParallel(modelF, device_ids=gpu_list)
    modelB = nn.DataParallel(modelB, device_ids=gpu_list)
    modelC = nn.DataParallel(modelC, device_ids=gpu_list)

    args.modelpath = args.output_dir_source + '/target_F.pt'   
    modelF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_source + '/target_B.pt'   
    modelB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_source + '/target_C.pt'   
    modelC.load_state_dict(torch.load(args.modelpath))

    modelF.eval()
    modelB.eval()
    modelC.eval()

    print("Models loaded")
    if args.da == 'oda':
        acc_os1, acc_os2, acc_unknown = cal_acc_oda(dataset_loaders['test'], modelF, modelB, modelC)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name, acc_os2, acc_os1, acc_unknown)
    else:
        if args.dataset=='visda-2017':
            acc, acc_list = cal_acc(dataset_loaders['test'], modelF, modelB, modelC, True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n' + acc_list
        else:
            acc, _, predict, idx = cal_acc(dataset_loaders['test'], modelF, modelB, modelC, False)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc)
    print(log_str)

    return acc, predict, idx

def print_args(args: argparse.Namespace) -> str:
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--source', type=int, default=0, help="source")
    parser.add_argument('--target', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument('--dataset', type=str, default='office-home', choices=['visda-2017', 'office', 'office-home', 'office-caltech', 'pacs', 'domain_net'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='vit', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='STDA_weights')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--bsp', type=bool, default=False)
    parser.add_argument('--se', type=bool, default=False)
    parser.add_argument('--nl', type=bool, default=False)
    parser.add_argument('--cls_par', type=float, default=0.2)

    args = parser.parse_args()

    if args.dataset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dataset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dataset =='domain_net':
        names = ['clipart', 'infograph', 'painting', 'quickdraw', 'sketch', 'real']
        args.class_num = 345

    gpu_id = ''
    for i in range(torch.cuda.device_count()):
        gpu_id += str(i) + ','
    gpu_id.removesuffix(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    args.batch_size = args.batch_size * torch.cuda.device_count()

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    folder = './data/'
    args.source_dataset_path = folder + args.dataset + '/' + names[args.source] + '.txt'
    args.test_dataset_path = folder + args.dataset + '/' + names[args.target] + '.txt'

    print(print_args(args))
    if args.dataset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.source_classes = [i for i in range(65)]
            args.target_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 25
            args.source_classes = [i for i in range(25)]
            args.target_classes = [i for i in range(65)]

    args.name_source = names[args.source][0].upper()
    args.save_dir = os.path.join('csv/', args.dataset)
    if not os.path.exists(args.save_dir):
        os.system('mkdir -p ' + args.save_dir)

    for i in range(len(names)):
        if i == args.source:
            continue

        args.target = i
        args.name = names[args.source][0].upper() + names[args.target][0].upper()

        args.output_dir_source = os.path.join(args.output, 'STDA', args.dataset, args.name.upper())
        args.source_dataset_path = folder + args.dataset + '/' + names[args.source] + '.txt'
        args.test_dataset_path = folder + args.dataset + '/' + names[args.target] + '.txt'

        accuracy, pseudo_label, idx = test_target(args)
        txt_test = open(args.test_dataset_path).readlines()
        img_pathes = []
        labels = []
        for i in list(idx):
            image_path, label = txt_test[i].split(' ')
            img_pathes.append(image_path)
            labels.append(label)
        dict = {'Domain': args.target, 'Image Path': img_pathes, 'Actual Label': labels, 'Pseudo Label': pseudo_label}
        df = pd.DataFrame(dict)
        df.to_csv(os.path.join(args.save_dir, names[args.source]+'.csv'), mode='a', header=False, index=False)