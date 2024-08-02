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

from lib.toolkit import print_argparse, cal_norm
from lib.wavUtils import DoNothing, Components
from CoNMix.lib.prepare_dataset import ExpandChannel, Dataset_Idx
from lib.datasets import load_from
from CoNMix.analysis import load_model, load_origin_stat
from CoNMix.STDA import build_optim, obtain_label, lr_scheduler, inference
from CoNMix.lib.loss import SoftCrossEntropyLoss, soft_CE
from CoNMix.lib.plr import plr

def build_dataset(args: argparse.Namespace) -> tuple[Dataset, Dataset, Dataset]:
    max_ms = 1000
    sample_rate = 16000
    n_mels=128
    hop_length=377
    meta_file_name = 'speech_commands_meta.csv'

    # test dataset build
    tf_array = [
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((224, 224), antialias=False),
        ]
    if args.normalized:
        print('test dataset mean and standard deviation calculation')
        test_dataset = load_from(root_path=args.test_dataset_root_path, index_file_name=meta_file_name, data_tf=Components(transforms=tf_array))
        test_mean, test_std = cal_norm(loader=DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, drop_last=False))
        tf_array.append(v_transforms.Normalize(mean=test_mean, std=test_std))
    test_dataset = load_from(root_path=args.test_dataset_root_path, index_file_name=meta_file_name, data_tf=Components(transforms=tf_array))

    # weak augmentation dataset build
    if args.data_type == 'final':
        tf_array = [
            # v_transforms.RandomHorizontalFlip(),
            DoNothing(),
        ]
    elif args.data_type == 'raw':
        tf_array = [
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            # a_transforms.FrequencyMasking(freq_mask_param=.02),
            # a_transforms.TimeMasking(time_mask_param=.02),
            ExpandChannel(out_channel=3),
            # v_transforms.Resize((256, 256), antialias=False),
            # v_transforms.RandomCrop(224)
            v_transforms.Resize((224, 224), antialias=False),
        ]
    else:
        raise Exception('No support')
    if args.normalized:
        print('weak augmentation mean and standard deviation calculation')
        weak_aug_dataset = load_from(root_path=args.weak_aug_dataset_root_path, index_file_name=meta_file_name, data_tf=Components(transforms=tf_array))
        weak_mean, weak_std = cal_norm(loader=DataLoader(dataset=weak_aug_dataset, batch_size=256, shuffle=False, drop_last=False))
        tf_array.append(v_transforms.Normalize(mean=weak_mean, std=weak_std))
    weak_aug_dataset = load_from(root_path=args.weak_aug_dataset_root_path, index_file_name=meta_file_name, data_tf=Components(transforms=tf_array))

    # strong augmentation dataset build
    if args.data_type == 'final':
        tf_array = [DoNothing()]
    elif args.data_type == 'raw':
        tf_array = [
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            a_transforms.FrequencyMasking(freq_mask_param=.1),
            a_transforms.TimeMasking(time_mask_param=.1),
            ExpandChannel(out_channel=3),
            # v_transforms.Resize((256, 256), antialias=False),
            # v_transforms.RandomCrop(224)
            v_transforms.Resize((224, 224), antialias=False),
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
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands'])
    ap.add_argument('--test_dataset_root_path', type=str)
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

    ap.add_argument('--corruption', type=str, choices=['doing_the_dishes', 'dude_miaowing', 'exercise_bike', 'pink_noise', 'running_tap', 'white_noise'])
    ap.add_argument('--severity_level', type=float, default=1.0)

    ap.add_argument('--seed', type=int, default=2024, help='random seed')
    ap.add_argument('--normalized', action='store_true')

    ap.add_argument('--max_epoch', type=int, default=100, help="max iterations")
    ap.add_argument('--interval', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=48, help="batch_size")
    # ap.add_argument('--test_batch_size', type=int, default=128, help="batch_size")
    ap.add_argument('--lr', type=float, default=1e-3, help="learning rate")

    ap.add_argument('--consist', type=bool, default=True, help='Consist loss -> soft cross-entropy loss')
    ap.add_argument('--fbnm', type=bool, default=True, help='fbnm -> Nuclear-norm Maximization loss')

    ap.add_argument('--threshold', type=int, default=0)
    ap.add_argument('--cls_par', type=float, default=0.2, help='lambda 2 | Pseudo-label loss capable')
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
    ap.add_argument('--plr', type=int, default=1, help='Pseudo-label refinement')
    ap.add_argument('--sdlr', type=int, default=1, help='lr_scheduler capable')

    args = ap.parse_args()
    args.class_num = 30
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
        project='Audio Classification CoNMix (STDA)', name=f'{args.dataset}_{args.severity_level}', mode='online' if args.wandb else 'disabled',
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
    for epoch in range(1, args.max_epoch+1):
        print(f'Epoch {epoch}/{args.max_epoch}')
        ttl_loss = 0.
        ttl_cls_loss = 0.
        ttl_const_loss = 0.
        ttl_fbnm_loss = 0.
        ttl_num = 0
        for weak_features, _, idxes in tqdm(weak_test_loader):
            batch_size = weak_features.shape[0]
            if iter % interval_iter == 0 and args.cls_par >= 0:
                modelF.eval()
                modelB.eval()
                modelC.eval()
                # print('Starting to find Pseudo Labels! May take a while :)')
                # test loader same as target but has 3*batch_size compared to target and train
                mem_label, soft_output, dd, mean_all_output, actual_label = obtain_label(loader=test_loader, modelF=modelF, modelB=modelB, modelC=modelC, args=args)

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
                classifier_loss = SoftCrossEntropyLoss(outputs[0:batch_size], pred)
                classifier_loss = args.cls_par*torch.mean(classifier_loss)
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
                    lr_scheduler(optimizer, iter_num=iter, max_iter=max_iter)

                accuracy = inference(modelF=modelF, modelB=modelB, modelC=modelC, data_loader=test_loader, device=args.device)
                if accuracy > max_accu:
                    max_accu = accuracy
                    torch.save(modelF.state_dict(), os.path.join(args.full_output_path, args.STDA_modelF_weight_file_name))
                    torch.save(modelB.state_dict(), os.path.join(args.full_output_path, args.STDA_modelB_weight_file_name))
                    torch.save(modelC.state_dict(), os.path.join(args.full_output_path, args.STDA_modelC_weight_file_name))
                wandb.log({'Accuracy/classifier accuracy': accuracy, 'Accuracy/max classifier accuracy': max_accu})
                ttl_loss = ttl_loss / ttl_num * 100.
                ttl_cls_loss = ttl_cls_loss / ttl_num * 100.
                ttl_const_loss = ttl_const_loss / ttl_num * 100.
                ttl_fbnm_loss = ttl_fbnm_loss / ttl_num * 100.
                wandb.log({"LOSS/total loss":ttl_loss, "LOSS/Pseudo-label cross-entorpy loss":ttl_cls_loss, "LOSS/consistency loss":ttl_const_loss, "LOSS/Nuclear-norm Maximization loss":ttl_fbnm_loss})
                modelF.train()
                modelB.train()
                modelC.train()