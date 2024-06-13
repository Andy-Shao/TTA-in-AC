import argparse
import os
import random
import numpy as np
import wandb
from typing import Tuple, Dict
from tqdm import tqdm
from timm.data.auto_augment import rand_augment_transform # timm for randaugment

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms

from lib import models
from helper.plr import plr
from lib.loss import SoftCrossEntropyLoss, Entropy, soft_CE
from helper.data_list import ImageList_idx

def strong_augment(resize_size=256, crop_size=224, alexnet=False) -> nn.Module:
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        assert False, 'Error'
    return transforms.Compose(transforms=[
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        rand_augment_transform(config_str='rand-m9-mstd0.5',hparams={'translate_const': 117}),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False) -> nn.Module:
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        assert False, 'Error'
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
        assert False, 'Error'
    return transforms.Compose(transforms=[
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def cal_acc(loader: DataLoader, modelF: nn.Module, modelB: nn.Module, modelC: nn.Module, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cuda()
            labels = data[1]
            outputs = modelC(modelB(modelF(inputs)))
            if start_test:
                all_output = outputs.float().numpy()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), dim=0)
                all_label = torch.cat((all_label, labels.float()), dim=0)
    _, predict = torch.max(all_output, dim=1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy * 100, mean_ent

def lr_scheduler(optimizer:optim.Optimizer, iter_num:int, max_iter:int, gamma=10, power=0.75) -> optim.Optimizer:
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = .9
        param_group['nesterov'] = True
    return optimizer

def obtain_label(loader: DataLoader, modelF: nn.Module, modelB: nn.Module, modelC: nn.Module, args: argparse.Namespace) -> Tuple:
    start_test = True
    # Accumulate feat, logint and gt labels
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in tqdm(range(len(loader))):
            data = next(iter_test)
            inputs = data[0].cuda()
            labels = data[1]
            features = modelB(modelF(inputs))
            outputs = modelC(features)
            if start_test:
                all_feature = features.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_feature = torch.cat((all_feature, features.float().cpu()), dim=0)
                all_output = torch.cat((all_output, outputs.float().cpu()), dim=0)
                all_label = torch.cat((all_label, labels.float()), dim=0)
    ##################### Done ##################################
    print('Clustering')
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
    wandb.log({"Pseudo_Label_Accuracy": acc*100})
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    dd = F.softmax(torch.from_numpy(dd), dim=1)
    return pred_label, all_output.cpu().numpy(), dd.numpy().astype('float32'), mean_all_output, all_label.cpu().numpy().astype(np.uint16)

def get_strong_aug(dataset: Dataset, idx: torch.Tensor) -> torch.Tensor:
    augment_img = torch.cat([dataset[i][0] for i in idx], dim=0)
    return augment_img

def op_copy(optimizer: optim.Optimizer) -> optim.Optimizer:
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def data_load(args: argparse.Namespace) -> Tuple[Dict, Dict]:
    datasets = {}
    dataset_loaders = {}
    train_batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    txt_target = open(args.target_dataset_path).readlines()
    txt_test = open(args.test_dataset_path).readlines()
    txt_eval_dn = open(args.txt_eval_dn).readlines()

    datasets['target'] = ImageList_idx(txt_target, transform=image_train())
    dataset_loaders['target'] = DataLoader(dataset=datasets['target'], batch_size=train_batch_size, shuffle=True, num_workers=args.worker, drop_last=True)
    datasets['test'] = ImageList_idx(txt_test, transform=image_test())
    dataset_loaders['test'] = DataLoader(dataset=datasets['test'], batch_size=test_batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
    datasets['strong_aug'] = ImageList_idx(txt_test, transform=strong_augment())
    dataset_loaders['strong_aug'] = DataLoader(dataset=datasets['strong_aug'], batch_size=test_batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
    if args.dataset == 'domain_net':
        datasets['eval_dn'] = ImageList_idx(txt_eval_dn, transform=image_train())
        dataset_loaders['eval_dn'] = DataLoader(dataset=datasets['eval_dn'], batch_size=test_batch_size, shuffle=False, num_workers=args.worker, drop_last=False)
    else:
        dataset_loaders['eval_dn'] = dataset_loaders['test']

    return dataset_loaders, datasets

def train_target(args: argparse.Namespace) -> Tuple[nn.Module, nn.Module, nn.Module]:
    dataset_loaders, datasets = data_load(args)
    if args.net[0:3] == 'res':
        modelF = models.ResBase(res_name=args.net, se=args.se, nl=args.nl).cuda()
    elif args.net[0:3] == 'vgg':
        modelF = models.VGGBase(vgg_name=args.net).cuda()
    elif args.net == 'vit':
        modelF = models.ViT().cuda()
    
    modelB = models.feat_bootleneck(type=args.classifier, feature_dim=modelF.in_features, bottleneck_dim=args.bottleneck).cuda()
    modelC = models.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    gpu_list = [i for i in range(torch.cuda.device_count())]
    print(f"Let's use {len(gpu_list)} GPUs")
    modelF = nn.DataParallel(modelF, device_ids=gpu_list)
    modelB = nn.DataParallel(modelB, device_ids=gpu_list)
    modelC = nn.DataParallel(modelC, device_ids=gpu_list)

    modelpath = args.output_dir_src + '/source_F.pt'
    modelF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    modelB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_srd + '/source_C.pt'
    modelC.load_state_dict(torch.load(modelpath))
    print('Model loaded!')

    param_group = []
    for k, v in modelF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params':v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in modelB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params':v, 'lr': args.lr * args.lr_decay2}]
        else: 
            v.requires_grad = False
    for k, v in modelC.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params':v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    
    optimizer = optim.SGD(params=param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dataset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    print('Training Started')
    max_accuracy = 0
    while iter_num < max_iter:
        try:
            inputs_test, target_index = next(iter_test)
        except:
            iter_test = iter(dataset_loaders['target'])
            inputs_test, _, target_index = next(iter_test)
            inputs_test_strong = get_strong_aug(datasets['strong_aug'], target_index)
    
        if inputs_test.size(0) == 1: 
            continue

        inputs_test_week = inputs_test.cuda()
        inputs_test_strong = inputs_test_strong.cuda()
        inputs_test = torch.cat([inputs_test_week,inputs_test_strong],dim=0)

        if (iter_num % interval_iter == 0 and args.cls_par >= 0):
            modelF.eval()
            modelB.eval()
            modelC.eval()
            print('Starting to find Pseudo Labels! May take a while :)')
            # test loader same as target but has 3*batch_size compared to target and train
            mem_label, soft_output, dd, mean_all_output, actual_label = obtain_label(dataset_loaders['test'], modelF, modelB, modelC, args)

            if args.plr: # Pseudo-label refinement
                if iter_num == 0:
                    prev_mem_label = mem_label
                    if args.soft_pl:
                        mem_label = dd
                else:
                    mem_label = plr(prev_mem_label, mem_label, dd, args.class_num, alpha = args.alpha)
                    if not args.soft_pl:
                        mem_label = mem_label.argmax(axis=1).astype(int)
                        refined_label = mem_label
                    else:
                        refined_label = mem_label.argmax(axis=1)
                    prev_mem_label = refined_label
            print('Completed finding Pseudo Labels\n')
            mem_label = torch.from_numpy(mem_label).cuda()
            dd = torch.from_numpy(dd).cuda()
            mean_all_output = torch.from_numpy(mean_all_output).cuda()

            modelF.train()
            modelB.train()
            modelC.train()
        
        iter_num += 1

        features = modelB(modelF(inputs_test))
        outputs = modelC(features)

        # Pseudo-label cross-entropy loss
        if args.cls_par > 0:
            with torch.no_grad():
                pred = mem_label[target_index]
            if args.soft_pl:
                classifier_loss = SoftCrossEntropyLoss(outputs[0:args.batch_size], pred)
                classifier_loss = torch.mean(classifier_loss)
            else:
                classifier_loss = nn.CrossEntropyLoss()(outputs[0:args.batch_size], pred)
            classifier_loss *= args.cls_par
        else:
            classifier_loss = torch.tensor(.0).cuda()

        # fbnm -> Nuclear-norm Maximization loss
        if args.fbnm:
            softmax_out = nn.Softmax(dim=1)(outputs)
            list_svd,_ = torch.sort(torch.sqrt(torch.sum(torch.pow(softmax_out,2),dim=0)), descending=True)
            fbnm_loss = - torch.mean(list_svd[:min(softmax_out.shape[0],softmax_out.shape[1])])
            fbnm_loss = args.fbnm_par*fbnm_loss
        else:
            fbnm_loss = torch.tensor(.0).cuda()

        # entropy loss (deprecated)
        if args.ent:
            # find number of psuedo sample per class for handling class imbalance for entropy maximization
            softmax_out = nn.Softmax(dim=1)(outputs)
            entorpy_loss = torch.mean(Entropy(soft_output))
            en_loss = entorpy_loss.item()
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                gen_loss = gentropy_loss.item()
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
        else:
            im_loss = torch.tensor(.0).cuda()
        
        # Consist loss -> soft cross-entropy loss
        if args.consist:
            softmax_out = nn.Softmax(dim=1)(outputs)
            expectation_ratio = mean_all_output/torch.mean(softmax_out[0:args.batch_size],dim=0)
            with torch.no_grad():
                soft_label_norm = torch.norm(softmax_out[0:args.batch_size]*expectation_ratio,dim=1,keepdim=True) #Frobenius norm
                soft_label = (softmax_out[0:args.batch_size]*expectation_ratio)/soft_label_norm
            consistency_loss = args.const_par*torch.mean(soft_CE(softmax_out[args.batch_size:],soft_label))
            cs_loss = consistency_loss.item()
        else: 
            consistency_loss = torch.tensor(.0).cuda()
        total_loss = classifier_loss + fbnm_loss + consistency_loss

        wandb.log({"total loss":total_loss.item(),"Pseudo-label cross-entorpy loss":classifier_loss.item(), "im_loss":im_loss.item(),"consistency loss":consistency_loss.item(), "Nuclear-norm Maximization loss":fbnm_loss.item()})

        optimizer.zero_grad()
        total_loss.backward()
        print(f'Task: {args.name}, Iter:{iter_num}/{max_iter} \t total loss {total_loss.item():.4f}')
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            if args.sdlr:
                lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

            modelF.eval()
            modelB.eval()
            modelC.eval()

            accuracy_eval_dn, _ = cal_acc(dataset_loaders["eval_dn"], modelF, modelB, modelC, False)
            if accuracy_eval_dn >= max_accuracy:
                max_accuracy = accuracy_eval_dn
                torch.save(modelF.state_dict(), os.path.join(args.output_dir, 'target_F.pt'))
                torch.save(modelB.state_dict(), os.path.join(args.output_dir, 'target_B.pt'))
                torch.save(modelC.state_dict(), os.path.join(args.output_dir, 'target_C.pt'))
                print('Model Saved!!!')
            wandb.log({"STDA_Test_Accuracy":accuracy_eval_dn, "Max_Acc": max_accuracy})
            log_str = '\nTask: {}, Iter:{}/{}; Final Eval test = {:.2f}%'.format(args.name, iter_num, max_iter, accuracy_eval_dn)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            if args.earlystop:
                print('Stopping Early!')
                return modelF, modelB, modelC

            modelF.train()
            modelB.train()
            modelC.train()
    
    print('Maximum Accuracy: ', max_accuracy)
    return modelF, modelB, modelC

def print_args(args: argparse.Namespace) -> str:
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    print(s)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rand-Augment')
    parser.add_argument('--source', type=int, default=0, help="source")
    parser.add_argument('--target', type=int, default=1, nargs='+', help="target")
    parser.add_argument('--max_epoch', type=int, default=100, help="max iterations")
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=48, help="batch_size")
    parser.add_argument('--test_batch_size', type=int, default=128, help="batch_size")
    parser.add_argument('--dataset', type=str, default='office-home')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', type=bool, default=False)
    parser.add_argument('--ent', type=bool, default=False)
    parser.add_argument('--kd', type=bool, default=False)
    parser.add_argument('--se', type=bool, default=False)
    parser.add_argument('--nl', type=bool, default=False)
    parser.add_argument('--consist', type=bool, default=True)
    parser.add_argument('--fbnm', type=bool, default=True)

    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.2, help='lambda 2')
    parser.add_argument('--alpha', type=float, default=0.9)

    parser.add_argument('--const_par', type=float, default=0.2, help='lambda 3')
    parser.add_argument('--ent_par', type=float, default=1.3)
    parser.add_argument('--fbnm_par', type=float, default=4.0, help='lambda 1')

    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='STDA_weights', help='Save ur weights here')
    parser.add_argument('--input_source', type=str, default='pre_train', help='Load SRC training wt path')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=False)
    parser.add_argument('--earlystop', type=int, default=0)
    parser.add_argument('--plr', type=int, default=1, help='Pseudo-label refinement')
    parser.add_argument('--soft_pl', type=int, default=1)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--worker', type=int, default=8)
    parser.add_argument('--wandb', type=int, default=1)
    parser.add_argument('--sdlr', type=int, default=1)

    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset =='domain_net':
        names = ['clipart', 'infograph', 'painting', 'quickdraw','sketch', 'real']
        args.class_num = 345

    gpu_id = ''
    for i in range(torch.cuda.device_count()):
        gpu_id += str(i) + ','
    gpu_id.removesuffix(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    args.batch_size = args.batch_size * torch.cuda.device_count()
    args.test_batch_size = args.test_batch_size * torch.cuda.device_count()

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if type(args.target) == int:
        args.target = [args.target]
    
    for i in args.target:
        if i == args.source:
            continue

        folder = './data/'
        args.source_dataset_path = folder + args.dataset + '/' + names[args.source] + '.txt'
        args.test_dataset_path = folder + args.dataset + '/' + names[i] + '.txt'
        args.target_dataset_path = folder + args.dataset + '/' + names[i] + '.txt'

        if args.dataset == 'domain_net':
            args.txt_eval_dn = folder + args.dataset + '/' + names[i] + '_test.txt'
        else:
            args.txt_eval_dn = args.target_dataset_path

        mode = 'online' if args.wandb else 'disable'
        wandb.init(project='CoNMix ECCV', name=f'STDA {names[args.source]} to {names[i]} '+args.suffix, reinit=True,mode=mode, config=args, tags=[args.dataset, args.net, 'STDA'])

        args.output_dir_source = os.path.join(args.input_srouce, args.da, args.dataset, names[args.source][0].upper())
        args.output_dir = os.path.join(args.output, 'STDA', args.dataset, names[args.source][0].upper() + names[args.target][0].upper())
        args.name = names[args.source][0].upper() + names[i][0].upper()

        if not os.path.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.out_file = open(os.path(args.output_dir, 'log.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)