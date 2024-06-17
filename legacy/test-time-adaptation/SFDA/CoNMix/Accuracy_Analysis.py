import argparse
import os
import wandb
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms

from helper.data_list import ImageList
from lib import models

def test(modelF: nn.Module, modelB: nn.Module, modelC: nn.Module, dataloader: DataLoader) -> tuple[float, float]:
    ttl_number = 0
    ttl_correct = 0

    modelF.eval()
    modelB.eval()
    modelC.eval()
    
    for feature, labels in dataloader:
        feature = feature.cuda()
        with torch.no_grad():
            outputs = modelC(modelB(modelF(feature)))
        _, preds = torch.max(outputs, dim=1)
        ttl_number += labels.size(0)
        ttl_correct += torch.sum(torch.squeeze(preds).cpu().float() == labels).item()

    return ttl_number, ttl_correct * 100. / ttl_number

def init_src_model_load(args: argparse.Namespace) -> tuple[nn.Module, nn.Module, nn.Module]:
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

def shifted_test(args: argparse.Namespace) -> tuple[float, float]:
    dataloader, dataset = load_data(args)

    modelF, modelB, modelC = init_src_model_load(args)

    weight_path = args.MTDA_weight_path + '/' + args.dataset + '/' + names[args.source]
    modelpath = weight_path + '/target_F.pt'
    modelF.load_state_dict(torch.load(modelpath))
    modelpath = weight_path + '/target_B.pt'
    modelB.load_state_dict(torch.load(modelpath))
    modelpath = weight_path + '/target_C.pt'
    modelC.load_state_dict(torch.load(modelpath))

    gpu_list = [i for i in range(torch.cuda.device_count())]
    print(f"Let's use {len(gpu_list)} GPUs")
    modelF = nn.DataParallel(modelF, device_ids=gpu_list)
    modelB = nn.DataParallel(modelB, device_ids=gpu_list)
    modelC = nn.DataParallel(modelC, device_ids=gpu_list)
    
    return test(modelF=modelF, modelB=modelB, modelC=modelC, dataloader=dataloader)

def origin_test(args: argparse.Namespace) -> tuple[float, float]:
    dataloader, dataset = load_data(args)

    if args.net[0:3] == 'res':
        modelF = models.ResBase(res_name=args.net, se=args.se, nl=args.nl).cuda()
    elif args.net[0:3] == 'vgg':
        modelF = models.VGGBase(vgg_name=args.net).cuda()
    elif args.net == 'vit':
        modelF = models.ViT().cuda()
    
    modelB = models.feat_bootleneck(type=args.classifier, feature_dim=modelF.in_features, bottleneck_dim=args.bottleneck).cuda()
    modelC = models.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    weight_path = args.origin_weight_path + '/' + args.da + '/' + args.dataset + '/' + names[args.source][0].upper()

    gpu_list = [i for i in range(torch.cuda.device_count())]
    print(f"Let's use {len(gpu_list)} GPUs")
    modelF = nn.DataParallel(modelF, device_ids=gpu_list)
    modelB = nn.DataParallel(modelB, device_ids=gpu_list)
    modelC = nn.DataParallel(modelC, device_ids=gpu_list)

    modelpath = weight_path + '/source_F.pt'
    modelF.load_state_dict(torch.load(modelpath))
    modelpath = weight_path + '/source_B.pt'
    modelB.load_state_dict(torch.load(modelpath))
    modelpath = weight_path + '/source_C.pt'
    modelC.load_state_dict(torch.load(modelpath))
    
    return test(modelF=modelF, modelB=modelB, modelC=modelC, dataloader=dataloader)

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

def load_data(args: argparse.Namespace) -> tuple[DataLoader, Dataset]:
    txt_test = open(args.test_dataset_label_path).readlines()

    dataset = ImageList(image_list=txt_test, transform=image_test())
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    return dataloader, dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Accuracy Analysis')
    parser.add_argument('--source', default=0, type=int)
    parser.add_argument('--origin_weight_path', default='pre_train', type=str, help='origin weight path')
    parser.add_argument('--MTDA_weight_path', default='MTDA_weights', type=str, help='After source free data adaptation weight path')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--dataset', type=str, default='office-home', choices=['visda-2017', 'office', 'office-home', 'office-caltech', 'pacs', 'domain_net'])
    parser.add_argument('--seed', default=2022, type=int, help='random seed')
    parser.add_argument('--wandb', type=int, default=1)
    parser.add_argument('--net', default="vit", type=str, help='model type (default: vit)')

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])

    args = parser.parse_args()
    gpu_id = ''
    for i in range(torch.cuda.device_count()):
        gpu_id += str(i) + ','
    gpu_id.removesuffix(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    args.batch_size = args.batch_size * torch.cuda.device_count()

    if args.seed != 0:
        torch.manual_seed(args.seed)

    if args.dataset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dataset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dataset =='domain_net':
        names = ['clipart', 'infograph', 'painting', 'quickdraw','sketch', 'real']
        args.class_num = 345

    mode = 'online' if args.wandb else 'disabled'
    wandb.init(project='CoNMix ECCV Analysis', name=f'Analysis {names[args.source]}', reinit=True,mode=mode, config=args, tags=[args.dataset, args.net, 'Analysis'])

    folder = './data/'
    df = pd.DataFrame(columns=['Source', 'Target', 'Model', 'TTL NUM', 'Original Accuracy', 'Adapted Accuracy'])
    for i in range(len(names)):
        if i == args.source:
            continue
        args.test_dataset_path = folder + args.dataset + '/' + names[i]
        args.test_dataset_label_path = folder + args.dataset + '/' + names[i] + '.txt'
        print(f'dataset path: {args.test_dataset_path}, label path: {args.test_dataset_label_path}')

        # Original model test
        # _, origin_accuracy = origin_test(args=args)
        origin_accuracy = 0.

        # Source free data adaptation test
        ttl_num, shifted_accuracy = shifted_test(args=args)

        df.loc[len(df)] = [names[args.source], names[i], args.net, ttl_num, origin_accuracy, shifted_accuracy]
        print(f'Source: {names[args.source]}, Target: {names[i]}, Model: {args.net}, TTL NUM: {ttl_num}, Original Accuracy: {origin_accuracy}, Adapted Accuracy: {shifted_accuracy}')
    wandb.log({'Test/accuracy': wandb.Table(dataframe=df)})
