import argparse
import wandb
import os
import random
import numpy as np
from tqdm import tqdm

import torch 
from torchvision import transforms as v_transforms
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from lib.toolkit import print_argparse, parse_mean_std, cal_norm
from lib.datasets import load_from
from CoNMix.analysis import load_model, load_origin_stat, inference
from CoNMix.pre_train import op_copy
from CoNMix.lib.prepare_dataset import Dataset_Idx, ExpandChannel
from lib.wavUtils import Components, DoNothing
from CoNMix.lib.loss import SoftCrossEntropyLoss, soft_CE
from CoNMix.lib.plr import plr

def inference(modelF: nn.Module, modelB: nn.Module, modelC: nn.Module, data_loader: DataLoader, device='cpu') -> float:
    modelF.eval()
    modelB.eval()
    modelC.eval()
    ttl_corr = 0.
    ttl_size = 0.
    for features, labels in tqdm(data_loader):
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            outputs = modelC(modelB(modelF(features)))
        _, preds = torch.max(outputs, dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]
    return ttl_corr / ttl_size * 100.

def lr_scheduler(optimizer:optim.Optimizer, iter_num:int, max_iter:int, gamma=10, power=0.75, weight_decay=1e-3) -> optim.Optimizer:
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = weight_decay
        param_group['momentum'] = .9
        param_group['nesterov'] = True
    return optimizer

def build_dataset(args: argparse.Namespace) -> tuple[Dataset, Dataset, Dataset]:
    if args.dataset == 'audio-mnist':
        max_ms=1000
        sample_rate=48000
        n_mels=128
        hop_length=377
        args.class_num = 10
        meta_file_name = 'audio_minst_meta.csv'
        if args.data_type == 'final':
            test_tf = [
                DoNothing(),
            ]
        elif args.data_type == 'raw':
            test_tf = [
                a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
                a_transforms.AmplitudeToDB(top_db=80),
                ExpandChannel(out_channel=3),
                v_transforms.Resize((256, 256), antialias=False),
                v_transforms.RandomCrop(224),
            ]
        else:
            raise Exception('No support')
        if args.normalized:
            print('calculate the test dataset mean and standard deviation')
            test_dataset = load_from(root_path=args.weak_aug_dataset_root_path, index_file_name=meta_file_name, data_tf=Components(transforms=test_tf))
            test_mean, test_std = cal_norm(loader=DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, drop_last=False))
            test_tf.append(v_transforms.Normalize(mean=test_mean, std=test_std))
        test_dataset = load_from(root_path=args.weak_aug_dataset_root_path, index_file_name=meta_file_name, data_tf=Components(transforms=test_tf))
        if args.data_type == 'final':
            weak_test_tf = [
                v_transforms.RandomHorizontalFlip(),
            ]
        elif args.data_type == 'raw':
            weak_test_tf = [
                a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
                a_transforms.AmplitudeToDB(top_db=80),
                ExpandChannel(out_channel=3),
                v_transforms.Resize((256, 256), antialias=False),
                v_transforms.RandomCrop(224),
                v_transforms.RandomHorizontalFlip(),
            ]
        else:
            raise Exception('No support')
        if args.normalized:
            print('calculate the weak dataset mean and standard deviation')
            weak_test_dataset = load_from(root_path=args.weak_aug_dataset_root_path, index_file_name=meta_file_name, data_tf=Components(transforms=weak_test_tf))
            weak_mean, weak_std = cal_norm(loader=DataLoader(dataset=weak_test_dataset, batch_size=256, shuffle=False, drop_last=False))
            weak_test_tf.append(v_transforms.Normalize(mean=weak_mean, std=weak_std))
        weak_test_dataset = load_from(root_path=args.weak_aug_dataset_root_path, index_file_name=meta_file_name, data_tf=Components(transforms=weak_test_tf))
        
        if args.data_type == 'final':
            strong_test_tf = [
                DoNothing(),
            ]
        elif args.data_type == 'raw':
            strong_test_tf = [
                a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
                a_transforms.AmplitudeToDB(top_db=80),
                ExpandChannel(out_channel=3),
                v_transforms.Resize((256, 256), antialias=False),
                v_transforms.RandomCrop(224),
            ]
        else:
            raise Exception('No support')
        if args.normalized:
            print('calculate the strong dataset mean and standard deviation')
            strong_test_dataset = load_from(root_path=args.strong_aug_dataset_root_path, index_file_name=meta_file_name, data_tf=Components(transforms=strong_test_tf))
            strong_mean, strong_std = cal_norm(loader=DataLoader(dataset=strong_test_dataset, batch_size=256, shuffle=False, drop_last=False))
            strong_test_tf.append(v_transforms.Normalize(mean=strong_mean, std=strong_std))
        strong_test_dataset = load_from(root_path=args.strong_aug_dataset_root_path, index_file_name=meta_file_name, data_tf=Components(transforms=strong_test_tf))
    else:
        raise Exception('No support')
    return test_dataset, weak_test_dataset, strong_test_dataset

def obtain_label(loader: DataLoader, modelF: nn.Module, modelB: nn.Module, modelC: nn.Module, args: argparse.Namespace, step:int) -> tuple:
    # Accumulate feat, logint and gt labels
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(args.device)
            features = modelB(modelF(inputs))
            outputs = modelC(features)
            if idx == 0:
                all_feature = features.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
            else:
                all_feature = torch.cat((all_feature, features.float().cpu()), dim=0)
                all_output = torch.cat((all_output, outputs.float().cpu()), dim=0)
                all_label = torch.cat((all_label, labels.float()), dim=0)
        inputs = None
        features = None
        outputs = None
    ##################### Done ##################################
    # print('Clustering')
    all_output = nn.Softmax(dim=1)(all_output)

    mean_all_output = torch.mean(all_output, dim=0).numpy()
    _, predict = torch.max(all_output, dim=1)

    # find accuracy on test sampels
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # find centroid per class
    if args.distance == 'cosine': 
        ######### Not Clear (looks like feature normalization though)#######
        all_feature = torch.cat((all_feature, torch.ones(all_feature.size(0), 1)), dim=1)
        all_feature = (all_feature.t() / torch.norm(all_feature, p=2, dim=1)).t() # here is L2 norm
    ### all_fea: extractor feature [bs,N]. all_feature is g_t in paper
    all_feature = all_feature.float().cpu().numpy()
    K = all_output.size(1) # number of classes
    aff = all_output.float().cpu().numpy() ### aff: softmax output [bs,c]

    # got the initial normalized centroid (k*(d+1))
    initc = aff.transpose().dot(all_feature)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

    cls_count = np.eye(K)[predict].sum(axis=0) # total number of prediction per class
    labelset = np.where(cls_count >= args.threshold) ### index of classes for which same sampeled have been detected # returns tuple
    labelset = labelset[0] # index of classes for which samples per class greater than threshold
    
    dd = all_feature @ initc[labelset].T # <g_t, initc>
    dd = np.exp(dd) # amplify difference
    pred_label = dd.argmax(axis=1) # predicted class based on the minimum distance
    pred_label = labelset[pred_label] # this will be the actual class
    
    for round in range(1): # calculate initc and pseduo label multi-times
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_feature)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = all_feature @ initc[labelset].T
        dd = np.exp(dd)
        pred_label = dd.argmax(axis=1)
        pred_label = labelset[pred_label]
    
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_feature)
    wandb.log({"Accuracy/Pseudo Label Accuracy": acc*100}, step=step)

    dd = F.softmax(torch.from_numpy(dd), dim=1)
    return pred_label, all_output.cpu().numpy(), dd.numpy().astype(np.float32), mean_all_output, all_label.cpu().numpy().astype(np.uint16)

def build_optim(args: argparse.Namespace, modelF: nn.Module, modelB: nn.Module, modelC: nn.Module) -> optim.Optimizer:
    param_group = []
    for _, v in modelF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params':v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for _, v in modelB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params':v, 'lr': args.lr * args.lr_decay2}]
        else: 
            v.requires_grad = False
    for _, v in modelC.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params':v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(params=param_group)
    optimizer = op_copy(optimizer)
    return optimizer

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Rand-Augment')
    ap.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--weak_aug_dataset_root_path', type=str)
    ap.add_argument('--strong_aug_dataset_root_path', type=str)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--modelF_weight_path', type=str)
    ap.add_argument('--modelB_weight_path', type=str)
    ap.add_argument('--modelC_weight_path', type=str)
    ap.add_argument('--STDA_modelF_weight_file_name', type=str)
    ap.add_argument('--STDA_modelB_weight_file_name', type=str)
    ap.add_argument('--STDA_modelC_weight_file_name', type=str)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--wandb_name', type=str, default='')
    ap.add_argument('--data_type', type=str, choices=['raw', 'final'], default='final')

    ap.add_argument('--corruption', type=str, default='gaussian_noise')
    ap.add_argument('--severity_level', type=float, default=.0025)
    ap.add_argument('--weak_corrupted_mean', type=str)
    ap.add_argument('--weak_corrupted_std', type=str)
    ap.add_argument('--strong_corrupted_mean', type=str)
    ap.add_argument('--strong_corrupted_std', type=str)

    ap.add_argument('--seed', type=int, default=2024, help='random seed')
    ap.add_argument('--normalized', action='store_true')

    ap.add_argument('--max_epoch', type=int, default=100, help="max iterations")
    ap.add_argument('--interval', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=48, help="batch_size")
    # ap.add_argument('--test_batch_size', type=int, default=128, help="batch_size")
    ap.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    ap.add_argument('--lr_gamma', type=int, default=10)
    ap.add_argument('--lr_weight_decay', type=float, default=1e-3)
    ap.add_argument('--cls_mode', type=str, default='logsoft_ce', choices=['logsoft_ce', 'logsoft_nll'])

    # ap.add_argument('--gent', type=bool, default=False)
    # ap.add_argument('--kd', type=bool, default=False)
    # ap.add_argument('--se', type=bool, default=False)
    # ap.add_argument('--nl', type=bool, default=False)
    ap.add_argument('--consist', type=bool, default=True, help='Consist loss -> soft cross-entropy loss')
    ap.add_argument('--fbnm', type=bool, default=True, help='fbnm -> Nuclear-norm Maximization loss')

    ap.add_argument('--threshold', type=int, default=0)
    ap.add_argument('--cls_par', type=float, default=1.0, help='lambda 2 | Pseudo-label loss capable')
    ap.add_argument('--alpha', type=float, default=0.9)

    ap.add_argument('--const_par', type=float, default=0.2, help='lambda 3')
    ap.add_argument('--ent_par', type=float, default=1.3)
    ap.add_argument('--fbnm_par', type=float, default=4.0, help='lambda 1')

    ap.add_argument('--lr_decay1', type=float, default=0.1)
    ap.add_argument('--lr_decay2', type=float, default=1.0)

    ap.add_argument('--bottleneck', type=int, default=256)
    ap.add_argument('--epsilon', type=float, default=1e-5)
    ap.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    ap.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    ap.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    # ap.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    # ap.add_argument('--issave', type=bool, default=False)
    ap.add_argument('--plr', type=int, default=1, help='Pseudo-label refinement')
    # ap.add_argument('--suffix', type=str, default='')
    # ap.add_argument('--worker', type=int, default=8)
    ap.add_argument('--sdlr', type=int, default=1, help='lr_scheduler capable')

    ap.add_argument('--early_stop', type=int, default=-1)
    ap.add_argument('--backup_weight', type=int, default=0)

    args = ap.parse_args()
    args.test_batch_size = args.batch_size * 3
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'CoNMix', 'STDA')
    try:
        os.makedirs(args.full_output_path)
    except:
        pass
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    #################################################

    wandb.init(
        project=f'Audio Classification CoNMix-STDA ({args.dataset})', 
        name=f'{args.corruption}_{args.severity_level}' if args.wandb_name == '' else args.wandb_name, 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'ViT']
    )

    test_dataset, weak_test_dataset, strong_test_dataset = build_dataset(args)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    weak_test_dataset = Dataset_Idx(dataset=weak_test_dataset)
    weak_test_loader = DataLoader(dataset=weak_test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    # build mode & load pre-train weight
    modelF, modelB, modelC = load_model(args)
    load_origin_stat(args, modelF=modelF, modelB=modelB, modelC=modelC)
    
    assert args.interval <= args.max_epoch, 'No support'
    optimizer = build_optim(args, modelF=modelF, modelB=modelB, modelC=modelC)
    max_iter = args.max_epoch * len(weak_test_loader)
    interval_iter = max_iter // args.interval
    iter = 0

    print('STDA Training Started')
    max_accu = 0.
    modelF.train()
    modelB.train()
    modelC.train()
    for epoch in range(1, args.max_epoch+1):
        if args.early_stop > 0 and epoch-1 == args.early_stop:
            break
        print(f'Epoch {epoch}/{args.max_epoch}')
        ttl_loss = 0.
        ttl_cls_loss = 0.
        ttl_const_loss = 0.
        ttl_fbnm_loss = 0.
        ttl_num = 0
        print('Training...')
        for weak_features, _, idxes in tqdm(weak_test_loader):
            batch_size = weak_features.shape[0]
            if iter % interval_iter == 0 and args.cls_par >= 0:
                modelF.eval()
                modelB.eval()
                modelC.eval()
                # print('Starting to find Pseudo Labels! May take a while :)')
                # test loader same as target but has 3*batch_size compared to target and train
                mem_label, soft_output, dd, mean_all_output, actual_label = obtain_label(loader=test_loader, modelF=modelF, modelB=modelB, modelC=modelC, args=args, step=epoch)

                if args.plr:
                    if iter == 0:
                        prev_mem_label = mem_label
                        mem_label = dd
                    else:
                        mem_label = plr(prev_mem_label, mem_label, dd, args.class_num, alpha = args.alpha)
                        prev_mem_label = mem_label.argmax(axis=1).astype(int)
                else:
                    mem_label = dd
    
                # print('Completed finding Pseudo Labels\n')
                mem_label = torch.from_numpy(mem_label).to(args.device)
                dd = torch.from_numpy(dd).to(args.device)
                mean_all_output = torch.from_numpy(mean_all_output).to(args.device)
                modelF.train()
                modelB.train()
                modelC.train()
            iter += 1

            strong_features = torch.cat([torch.unsqueeze(strong_test_dataset[idx][0], dim=0) for idx in idxes], dim=0)
            features = torch.cat([weak_features, strong_features], dim=0).to(args.device)

            outputs_B = modelB(modelF(features))
            outputs = modelC(outputs_B)

            # Pseudo-label cross-entropy loss
            if args.cls_par > 0:
                with torch.no_grad():
                    pred = mem_label[idxes]
                if args.cls_mode == 'logsoft_ce':
                    classifier_loss = SoftCrossEntropyLoss(outputs[0:batch_size], pred)
                    classifier_loss = torch.mean(classifier_loss)
                elif args.cls_mode == 'logsoft_nll':
                    softmax_output = nn.LogSoftmax(dim=1)(outputs[0:batch_size])
                    with torch.no_grad():
                        _, pred = torch.max(pred, dim=1)
                    classifier_loss = nn.NLLLoss(reduction='mean')(softmax_output, pred)
                classifier_loss = args.cls_par*classifier_loss
            else:
                classifier_loss = torch.tensor(.0).cuda()

            # fbnm -> Nuclear-norm Maximization loss
            if args.fbnm_par > 0:
                softmax_output = nn.Softmax(dim=1)(outputs)
                list_svd,_ = torch.sort(torch.sqrt(torch.sum(torch.pow(softmax_output,2),dim=0)), descending=True)
                fbnm_loss = - torch.mean(list_svd[:min(softmax_output.shape[0],softmax_output.shape[1])])
                fbnm_loss = args.fbnm_par*fbnm_loss
            else:
                fbnm_loss = torch.tensor(.0).cuda()
            
            # Consist loss -> soft cross-entropy loss
            if args.const_par > 0:
                softmax_output = nn.Softmax(dim=1)(outputs)
                expectation_ratio = mean_all_output/torch.mean(softmax_output[0:batch_size],dim=0)
                with torch.no_grad():
                    soft_label_norm = torch.norm(softmax_output[0:batch_size]*expectation_ratio,dim=1,keepdim=True) #Frobenius norm
                    soft_label = (softmax_output[0:batch_size]*expectation_ratio)/soft_label_norm
                consistency_loss = args.const_par*torch.mean(soft_CE(softmax_output[batch_size:],soft_label))
                cs_loss = consistency_loss.item()
            else:
                consistency_loss = torch.tensor(.0).cuda()
            total_loss = classifier_loss + fbnm_loss + consistency_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            ttl_loss += total_loss.item()
            ttl_cls_loss += classifier_loss.item()
            ttl_const_loss += consistency_loss.item()
            ttl_fbnm_loss += fbnm_loss.item()
            ttl_num += weak_features.shape[0]

            if iter % interval_iter == 0 or iter == max_iter:
                if args.sdlr:
                    lr_scheduler(optimizer, iter_num=iter, max_iter=max_iter, gamma=args.lr_gamma, weight_decay=args.lr_weight_decay)

        print('Inferencing...')
        accuracy = inference(modelF=modelF, modelB=modelB, modelC=modelC, data_loader=test_loader, device=args.device)
        if accuracy > max_accu:
            max_accu = accuracy
            if args.backup_weight == 0:
                torch.save(modelF.state_dict(), os.path.join(args.full_output_path, args.STDA_modelF_weight_file_name))
                torch.save(modelB.state_dict(), os.path.join(args.full_output_path, args.STDA_modelB_weight_file_name))
                torch.save(modelC.state_dict(), os.path.join(args.full_output_path, args.STDA_modelC_weight_file_name))
        wandb.log({'Accuracy/classifier accuracy': accuracy, 'Accuracy/max classifier accuracy': max_accu}, step=epoch)
        wandb.log({
            "LOSS/total loss":ttl_loss/ttl_num*100., 
            "LOSS/Pseudo-label cross-entorpy loss":ttl_cls_loss/ttl_num*100., 
            "LOSS/consistency loss":ttl_const_loss/ttl_num*100., 
            "LOSS/Nuclear-norm Maximization loss":ttl_fbnm_loss/ttl_num*100.
        }, step=epoch, commit=True)
        modelF.train()
        modelB.train()
        modelC.train()