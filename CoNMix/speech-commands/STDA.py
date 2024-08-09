import argparse
import random 
import numpy as np
import os
import wandb
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as a_transforms
from torchvision import transforms as v_transforms
import torch.nn as nn
from torch import optim

from lib.toolkit import print_argparse, cal_norm
from lib.wavUtils import DoNothing, Components
from CoNMix.lib.prepare_dataset import ExpandChannel, Dataset_Idx
from lib.datasets import load_from
from CoNMix.analysis import load_model, load_origin_stat
from CoNMix.STDA import build_optim, lr_scheduler
from CoNMix.lib.loss import SoftCrossEntropyLoss, soft_CE, Entropy
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
    # labelset == [0, 1, 2, ..., 29]
    
    dd = all_feature @ initc[labelset].T # <g_t, initc>
    dd = np.exp(dd) # amplify difference
    pred_label = dd.argmax(axis=1) # predicted class based on the minimum distance
    pred_label = labelset[pred_label] # this will be the actual class
    
    for round in range(args.initc_num): # calculate initc and pseduo label multi-times
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_feature)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = all_feature @ initc[labelset].T
        dd = np.exp(dd)
        pred_label = dd.argmax(axis=1)
        pred_label = labelset[pred_label]
    
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_feature)
    wandb.log({"Accuracy/Pseudo Label Accuracy": acc*100}, step=step)

    dd = nn.functional.softmax(torch.from_numpy(dd), dim=1)
    return pred_label, all_output.cpu().numpy(), dd.numpy().astype(np.float32), mean_all_output, all_label.cpu().numpy().astype(np.uint16)

def build_dataset(args: argparse.Namespace) -> tuple[Dataset, Dataset, Dataset]:
    max_ms = 1000
    sample_rate = 16000
    n_mels=129
    hop_length=125
    meta_file_name = 'speech_commands_meta.csv'

    # test dataset build
    if args.data_type == 'final':
        tf_array = [DoNothing()]
    elif args.data_type == 'raw':
        tf_array = [
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((224, 224), antialias=False),
        ]
    if args.normalized:
        print('test dataset mean and standard deviation calculation')
        test_dataset = load_from(root_path=args.weak_aug_dataset_root_path, index_file_name=meta_file_name, data_tf=Components(transforms=tf_array))
        test_mean, test_std = cal_norm(loader=DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, drop_last=False))
        tf_array.append(v_transforms.Normalize(mean=test_mean, std=test_std))
    test_dataset = load_from(root_path=args.weak_aug_dataset_root_path, index_file_name=meta_file_name, data_tf=Components(transforms=tf_array))

    # weak augmentation dataset build
    weak_aug_dataset = load_from(root_path=args.weak_aug_dataset_root_path, index_file_name=meta_file_name, data_tf=Components(transforms=tf_array))

    # strong augmentation dataset build
    if args.data_type == 'final':
        tf_array = [DoNothing()]
    elif args.data_type == 'raw':
        tf_array = [
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            # a_transforms.FrequencyMasking(freq_mask_param=.1),
            # a_transforms.TimeMasking(time_mask_param=.1),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((256, 256), antialias=False),
            v_transforms.RandomCrop(224)
        ]
    else:
        raise Exception('No support')
    if args.normalized:
        print('strong augmentation mean and standard deviation calculation')
        strong_aug_dataset = load_from(root_path=args.strong_aug_dataset_root_path, index_file_name=meta_file_name, data_tf=Components(transforms=tf_array))
        strong_mean, strong_std = cal_norm(loader=DataLoader(dataset=strong_aug_dataset, batch_size=256, shuffle=False, drop_last=False))
        tf_array.append(v_transforms.Normalize(mean=strong_mean, std=strong_std))
    strong_aug_dataset = load_from(root_path=args.strong_aug_dataset_root_path, index_file_name=meta_file_name, data_tf=Components(transforms=tf_array))

    return test_dataset, weak_aug_dataset, strong_aug_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Rand-Augment')
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands', 'speech-commands-purity'])
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
    ap.add_argument('--data_type', type=str, choices=['raw', 'final'], default='final')

    ap.add_argument('--corruption', type=str, choices=['doing_the_dishes', 'dude_miaowing', 'exercise_bike', 'pink_noise', 'running_tap', 'white_noise', 'gaussian_noise'])
    ap.add_argument('--severity_level', type=float, default=1.0)

    ap.add_argument('--seed', type=int, default=2024, help='random seed')
    ap.add_argument('--normalized', action='store_true')

    ap.add_argument('--max_epoch', type=int, default=100, help="max iterations")
    ap.add_argument('--interval', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=48, help="batch_size")
    ap.add_argument('--test_batch_size', type=int, default=128, help="batch_size")
    ap.add_argument('--lr', type=float, default=1e-3, help="learning rate")

    ap.add_argument('--consist', type=bool, default=True, help='Consist loss -> soft cross-entropy loss')
    ap.add_argument('--fbnm', type=bool, default=True, help='fbnm -> Nuclear-norm Maximization loss')

    ap.add_argument('--threshold', type=int, default=0)
    ap.add_argument('--cls_par', type=float, default=0.2, help='lambda 2 | Pseudo-label loss capable')
    ap.add_argument('--cls_mode', type=str, default='soft_ce', choices=['logsoft_ce', 'soft_ce', 'logsoft_nll'])
    ap.add_argument('--alpha', type=float, default=0.9)
    ap.add_argument('--const_par', type=float, default=0.2, help='lambda 3')
    ap.add_argument('--ent_par', type=float, default=1.3)
    ap.add_argument('--fbnm_par', type=float, default=4.0, help='lambda 1')
    ap.add_argument('--ent', action='store_true')
    ap.add_argument('--gent', action='store_true')

    ap.add_argument('--lr_decay1', type=float, default=0.1)
    ap.add_argument('--lr_decay2', type=float, default=1.0)

    ap.add_argument('--bottleneck', type=int, default=256)
    ap.add_argument('--epsilon', type=float, default=1e-5)
    ap.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    ap.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    ap.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    ap.add_argument('--plr', type=int, default=1, help='Pseudo-label refinement')
    ap.add_argument('--sdlr', type=int, default=1, help='lr_scheduler capable')
    ap.add_argument('--initc_num', type=int, default=1)

    args = ap.parse_args()
    if args.dataset == 'speech-commands':
        args.class_num = 30
        args.dataset_type = 'all'
    elif args.dataset == 'speech-commands-purity':
        args.class_num = 10
        args.dataset_type = 'commands'
    # args.test_batch_size = args.batch_size * 3
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
        project=f'Audio Classification CoNMix-STDA ({args.dataset})', name=f'{args.corruption}_{args.severity_level}', mode='online' if args.wandb else 'disabled',
        config=args, tags=['Audio Classification', args.dataset, 'ViT'])
    
    test_dataset, weak_test_dataset, strong_test_dataset = build_dataset(args)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    weak_test_dataset = Dataset_Idx(dataset=weak_test_dataset)
    weak_test_loader = DataLoader(dataset=weak_test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # build mode & load pre-train weight
    modelF, modelB, modelC = load_model(args)
    load_origin_stat(args, modelF=modelF, modelB=modelB, modelC=modelC)

    optimizer = build_optim(args, modelF=modelF, modelB=modelB, modelC=modelC)
    max_iter = args.max_epoch * len(weak_test_loader)
    interval_iter = max_iter // args.interval
    iter = 0

    print('STDA Training Started')
    modelF.train()
    modelB.train()
    modelC.train()
    max_accu = 0.
    for epoch in range(args.max_epoch):
        print(f'Epoch {epoch+1}/{args.max_epoch}')
        ttl_loss = 0.
        ttl_cls_loss = 0.
        ttl_const_loss = 0.
        ttl_fbnm_loss = 0.
        ttl_im_loss = 0.
        ttl_num = 0
        epoch_flag = True
        print('Training...')
        for weak_features, _, idxes in tqdm(weak_test_loader):
            batch_size = weak_features.shape[0]
            if epoch_flag and args.cls_par >= 0:
                epoch_flag = False
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
                elif args.cls_mode == 'soft_ce':
                    softmax_output = nn.Softmax(dim=1)(outputs[0:batch_size])
                    classifier_loss = nn.CrossEntropyLoss()(softmax_output, pred)
                elif args.cls_mode == 'logsoft_nll':
                    softmax_output = nn.LogSoftmax(dim=1)(outputs[0:batch_size])
                    _, pred = torch.max(pred, dim=1)
                    classifier_loss = nn.NLLLoss(reduction='mean')(softmax_output, pred)
                classifier_loss = args.cls_par * classifier_loss
            else:
                classifier_loss = torch.tensor(.0).cuda()

            # fbnm -> Nuclear-norm Maximization loss
            if args.fbnm:
                softmax_output = nn.Softmax(dim=1)(outputs)
                list_svd,_ = torch.sort(torch.sqrt(torch.sum(torch.pow(softmax_output,2),dim=0)), descending=True)
                fbnm_loss = - torch.mean(list_svd[:min(softmax_output.shape[0],softmax_output.shape[1])])
                fbnm_loss = args.fbnm_par*fbnm_loss
            else:
                fbnm_loss = torch.tensor(.0).cuda()

            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs)		# find number of psuedo sample per class for handling class imbalance for entropy maximization
                entropy_loss = torch.mean(Entropy(softmax_out))#softmax_outputs_stg = nn.Softmax(dim=1)(outputs_stg)
                #entropy_loss = torch.mean(loss.soft_CE(softmax_outputs_stg,gt_w))
                en_loss = entropy_loss.item()
                #entropy_loss = dist_loss(outputs_test, outputs_test,T=1.0)
                #entropy_loss = torch.mean(loss.Entropy(softmax_out))
                if args.gent:
                    #softmax_out = nn.Softmax(dim=1)(outputs)
                    msoftmax = softmax_out.mean(dim=0)
                    #msoftmax_stg = softmax_outputs_stg.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                    gen_loss = gentropy_loss.item()
                    entropy_loss -= gentropy_loss
                #m = 0.9*np.sin(np.minimum(np.pi/2,np.pi*iter_num/max_iter))
                im_loss = entropy_loss * args.ent_par
                #print("cls loss:{} en loss:{} gen loss:{} im_loss:{}".format(classifier_loss.item(), en_loss, gen_loss, im_loss.item()))
                #im_loss = entropy_loss * m
            else:
                im_loss = torch.tensor(0.0).cuda()
            
            # Consist loss -> soft cross-entropy loss
            if args.consist:
                softmax_output = nn.Softmax(dim=1)(outputs)
                expectation_ratio = mean_all_output/torch.mean(softmax_output[0:batch_size],dim=0)
                with torch.no_grad():
                    soft_label_norm = torch.norm(softmax_output[0:batch_size]*expectation_ratio,dim=1,keepdim=True) #Frobenius norm
                    soft_label = (softmax_output[0:batch_size]*expectation_ratio)/soft_label_norm
                consistency_loss = args.const_par*torch.mean(soft_CE(softmax_output[batch_size:],soft_label))
                cs_loss = consistency_loss.item()
            else:
                consistency_loss = torch.tensor(.0).cuda()
            total_loss = classifier_loss + fbnm_loss + im_loss + consistency_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            ttl_loss += total_loss.item()
            ttl_cls_loss += classifier_loss.item()
            ttl_const_loss += consistency_loss.item()
            ttl_fbnm_loss += fbnm_loss.item()
            ttl_im_loss += im_loss.item()
            ttl_num += weak_features.shape[0]

            if iter % interval_iter == 0:
                if args.sdlr:
                    lr_scheduler(optimizer, iter_num=iter, max_iter=max_iter, gamma=30)

        print('Inferecing...')
        accuracy = inference(modelF=modelF, modelB=modelB, modelC=modelC, data_loader=test_loader, device=args.device)
        if accuracy > max_accu:
            max_accu = accuracy
            torch.save(modelF.state_dict(), os.path.join(args.full_output_path, args.STDA_modelF_weight_file_name))
            torch.save(modelB.state_dict(), os.path.join(args.full_output_path, args.STDA_modelB_weight_file_name))
            torch.save(modelC.state_dict(), os.path.join(args.full_output_path, args.STDA_modelC_weight_file_name))
        wandb.log({'Accuracy/classifier accuracy': accuracy, 'Accuracy/max classifier accuracy': max_accu}, step=epoch)
        wandb.log({
            "LOSS/total loss":ttl_loss / ttl_num * 100., 
            "LOSS/Pseudo-label cross-entorpy loss":ttl_cls_loss / ttl_num * 100., 
            "LOSS/consistency loss":ttl_const_loss / ttl_num * 100., 
            "LOSS/Nuclear-norm Maximization loss":ttl_fbnm_loss / ttl_num * 100.,
            "LOSS/IM loss":ttl_im_loss / ttl_num * 100.,
        }, step=epoch)
        modelF.train()
        modelB.train()
        modelC.train()