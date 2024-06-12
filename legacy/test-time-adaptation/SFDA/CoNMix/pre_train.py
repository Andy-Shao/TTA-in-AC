import argparse
import os
import numpy as np
import random
from typing import Dict
import wandb
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from lib import models
from lib.loss import CrossEntropyLabelSmooth, Entropy
from helper.data_list import ImageList, ImageList_idx

def cal_acc_oda(loader: Dict, modelF: nn.Module, modelB: nn.Module, modelC: nn.Module):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cuda()
            labels = data[1]
            outputs = modelC(modelB(modelF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat(tensors=(all_output, outputs.float().cpu()), dim=0)
                all_label = torch.cat(tensors=(all_label, labels.float()), dim=0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(input=all_output, dim=1)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()
    initc = np.array([[0], [1]])
    kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1, 1))
    threshold = (kmeans.cluster_centers_).mean()
    
    predict[ent > threshold] = args.class_num
    matrix = confusion_matrix(y_true=all_label, y_pred=torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int),:]

    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()

    return np.mean(acc[:-1]), np.mean(acc), unknown_acc

def cal_acc(loader: Dict, modelF: nn.Module, modelB: nn.Module, modelC: nn.Module, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in tqdm(range(len(loader))):
            data = next(iter_test)
            inputs = data[0].cuda()
            labels = data[1]
            outputs = modelC(modelB(modelF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else: 
                all_output = torch.cat((all_output, outputs.float().cpu()), dim=0)
                all_label = torch.cat((all_label, labels.float()), dim=0)
            
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(input=all_output, dim=1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(y_true=all_label, y_pred=torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i,decimals=2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent

def lr_scheduler(optimizer: torch.optim.Optimizer, iter_num: int, max_iter: int, gamma=10, power=0.75) -> optim.Optimizer:
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = .9
        param_group['nestenv'] = True
        wandb.log({'MISC/LR': param_group['lr']})
    return optimizer

def test_target(args: argparse.Namespace):
    dataset_loaders = load_data(args=args)
    ## set base model
    if args.model[0:3] == 'res':
        modelF = models.ResBase(res_name=args.model,se=args.se,nl=args.nl).cuda()
    elif args.model[0:3] == 'vgg':
        modelF = models.VGGBase(vgg_name=args.model).cuda()
    elif args.model == 'vit':
        modelF = models.ViT().cuda()
    elif args.model[0:4] == 'deit':
        if args.model == 'deit_s':
            modelF = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).cuda()
        elif args.model == 'deit_b':
            modelF = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).cuda()
        modelF.in_features = 1000

    modelB = models.feat_bootleneck(type=args.classifier, feature_dim=modelF.in_features, bottleneck_dim=args.bottleneck).cuda()
    modelC = models.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    gpu_list = [i for i in range(torch.cuda.device_count())]
    print(f"Let's use {len(gpu_list)} GPUs for test")
    modelF = nn.DataParallel(modelF, device_ids=gpu_list)
    modelB = nn.DataParallel(modelB, device_ids=gpu_list)
    modelC = nn.DataParallel(modelC, device_ids=gpu_list)

    args.modelpath = args.output_dir_src + '/source_F.pt'
    modelF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B.pt'
    modelB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'
    modelC.load_state_dict(torch.load(args.modelpath))
    modelF.eval()
    modelB.eval()
    modelC.eval()

    if args.da == 'oda':
        accuracy_os1, accuracy_os2, accuracy_unknown = cal_acc_oda(dataset_loaders['test'], modelF, modelB, modelC)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name, accuracy_os1, accuracy_os2, accuracy_unknown)
    else: 
        if args.dataset == 'visda-2017':
            accuracy, accuracy_list = cal_acc(dataset_loaders['test'], modelF, modelB, modelC)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, accuracy) + '\n' + accuracy_list
        else:
            accuracy, _ = cal_acc(dataset_loaders['eval_dn'], modelF, modelB, modelC, False)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, accuracy)
    
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

def image_train(resize_size=256, crop_size=224, alexnet=False) -> nn.Module:
    if not alexnet:
        normalize = transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
    else: 
        assert False, 'No Support Operation!'
    return transforms.Compose(transforms=[
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False) -> nn.Module:
    if not alexnet:
        normalize = transforms.Normalize(mean=[.485, .456, .406], std=[.229,.224,.225])
    else:
        # normalize = 
        assert False, 'No Support Operation!'
    return transforms.Compose(transforms=[
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def load_data(args: argparse.Namespace) -> Dict:
    ## prepare data
    datasets = {}
    dataset_loaders = {}
    train_batch_size = args.batch_size
    
    text_source = open(args.source_dataset_path).readlines()
    text_test = open(args.test_dataset_path).readlines()

    if args.dataset == 'domain_net':
        txt_eval_dn = open(args.txt_eval_dn).readlines()
        print(f'Data samples: {len(txt_eval_dn)}')

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.source_classes[i]] = i 

        new_source = []
        for i in range(len(text_source)):
            rec = text_source[i].strip().split(' ')
            feature_path, label = rec[0], rec[1]
            if int(label) in args.srouce_classes:
                line = feature_path + ' ' + str(label_map_s[int(label)]) + '\n'
                new_source.append(line)
        text_source = new_source.copy()

        new_target = []
        for i in range(len(text_test)):
            rec = text_test.strip().split(' ')
            feature_path, label = rec[0], rec[1]
            if int(label) in args.target_classes:
                if int(label) in args.source_classes:
                    line = feature_path + ' ' + str(label_map_s[int(label)]) + '\n'
                    new_target.append(line)
                else:
                    line = feature_path + ' ' + str(len(label_map_s)) + '\n'
                    new_target.append(line)
        text_test = new_target.copy()
    
    if args.trte == 'val':
        data_size = len(text_source)
        tr_size = int(.9 * data_size)
        tr_text, te_text = torch.utils.data.random_split(text_source, [tr_size, data_size - tr_size])
    else: 
        data_size = len(text_source)
        tr_size = int(.9 * data_size)
        _, te_text = torch.utils.data.random_split(text_source, [tr_size, data_size - tr_size])
        tr_text = text_source

    datasets['source_tr'] = ImageList(tr_text, transform=image_train())
    dataset_loaders['source_tr'] = DataLoader(dataset=datasets['source_tr'], batch_size=train_batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
    datasets['source_te'] = ImageList(te_text, transform=image_test())
    dataset_loaders['source_te'] = DataLoader(dataset=datasets['source_tr'], batch_size=train_batch_size, shuffle=False, num_workers=args.worker, drop_last=False)

    datasets['test'] = ImageList(text_test, transform=image_test())
    dataset_loaders['test'] = DataLoader(dataset=datasets['test'], batch_size=train_batch_size, shuffle=False, num_workers=args.worker, drop_last=False)

    if args.dataset == 'domain_net':
        datasets['eval_dn'] = ImageList_idx(txt_eval_dn, transform=image_train())
        dataset_loaders['eval_dn'] = DataLoader(dataset=datasets['eval_dn'], batch_size=train_batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
    else: 
        dataset_loaders['eval_dn'] = dataset_loaders['test']
    
    return dataset_loaders

def op_copy(optimizer: optim.Optimizer) -> optim.Optimizer:
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def train_source(args: argparse.Namespace):
    dataset_loaders = load_data(args)
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

    accuracy_init = 0
    max_iter = args.max_epoch * len(dataset_loaders['source_tr'])
    interval_iter = max_iter // args.interval
    iter_num = 0

    modelF.train()
    modelB.train()
    modelC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except: 
            iter_source = iter(dataset_loaders['source_tr'])
            inputs_source, labels_source = next(iter_source)
        
        if inputs_source.size(0) == 1:
            continue
        
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = modelC(modelB(modelF(inputs_source)))

        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)
        wandb.log({'SRC Train: train_classifier_loss': classifier_loss.item()})

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            lr_scheduler(optimizer=optimizer, iter_num=iter_num, max_iter=max_iter)

            modelF.eval()
            modelB.eval()
            modelC.eval()
            if args.dataset == 'visda-2017':
                accuracy_s_te, accuracy_list = cal_acc(dataset_loaders['source_te'], modelF, modelB, modelC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, accuracy_s_te) + '\n' + accuracy_list
            else:
                accuracy_s_te, _ = cal_acc(dataset_loaders['source_te'], modelF, modelB, modelC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, accuracy_s_te)
            wandb.log({'SRC TRAIN: Acc' : accuracy_s_te})
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            if accuracy_s_te >= accuracy_init:
                accuracy_init = accuracy_s_te
                best_modelF = modelF.state_dict()
                best_modelB = modelB.state_dict()
                best_modelC = modelC.state_dict()

                torch.save(best_modelF, os.path.join(args.output_dir_src, 'source_F.pt'))
                torch.save(best_modelB, os.path.join(args.output_dir_src, 'source_B.pt'))
                torch.save(best_modelC, os.path.join(args.output_dir_src, 'source_C.pt'))
                print('Model Saved!!')
        
            modelF.train()
            modelB.train()
            modelC.train()
        iter_num += 1
    return modelF, modelB, modelC

def print_args(args: argparse.Namespace) -> str:
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
    args.source_dataset_path = folder + '/' + args.dataset + '/' + names[args.source] + '.txt'
    args.test_dataset_path = folder + '/' + args.dataset + '/' + names[args.target] + '.txt'

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
        args.source_dataset_path = folder + '/' + args.dataset + '/' + names[args.source] + '.txt'
        args.test_dataset_path = folder + '/' + args.dataset + '/' + names[args.target] + '.txt'
        if args.dataset == 'domain_net':
            args.txt_eval_dn = folder + '/' + args.dataset + '/' + names[args.target] + '_test.txt'
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