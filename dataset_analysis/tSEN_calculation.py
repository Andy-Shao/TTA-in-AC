import argparse
import os

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as v_transforms

from lib.datasets import FilterAudioMNIST, ClipDataset
from lib.scDataset import SpeechCommandsDataset
from lib.wavUtils import pad_trunc
from dataset_analysis.lib.tSEN_utils import cal_tSNE, inverse_dict, cal_tSNEs
from lib.toolkit import print_argparse

if __name__ == '__main__':
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--dataset', type=str, default=['audio-mnist', 'speech-commands', 'speech-commands-numbers', 'speech-commands-random', 'cifar-10'])
    arg_parse.add_argument('--dataset_root_path', type=str)
    arg_parse.add_argument('--num_workers', type=int, default=16)
    arg_parse.add_argument('--output_root_path', type=str, default='./result')
    arg_parse.add_argument('--batch_size', type=int, default=256)
    arg_parse.add_argument('--output_file', type=str)
    arg_parse.add_argument('--mode', type=str, default=['train', 'test', 'full'])
    arg_parse.add_argument('--rate', type=float, default=1.0)
    arg_parse.add_argument('--no_reduce', action='store_true')

    args = arg_parse.parse_args()
    output_full_root_path = os.path.join(args.output_root_path, 'dataset_analysis', args.dataset)
    try:
        if not os.path.exists(output_full_root_path):
            os.makedirs(output_full_root_path)
    except:
        pass

    print_argparse(args=args)
    #################################################

    if args.dataset == 'audio-mnist':
        label_dict = {
            0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 
            5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine'
        }
        
        if args.mode == 'full':
            train_dataset = FilterAudioMNIST(
                root_path=args.dataset_root_path,
                filter_fn=lambda x: x['accent'] == 'German',
                data_tsf=pad_trunc(max_ms=1000, sample_rate=48000),
                include_rate=False
            )
            test_dataset = FilterAudioMNIST(
                root_path=args.dataset_root_path,
                filter_fn=lambda x: x['accent'] != 'German',
                data_tsf=pad_trunc(max_ms=1000, sample_rate=48000),
                include_rate=False
            )
            ignore_test_clip = False
        else:
            dataset = FilterAudioMNIST(
                root_path=args.dataset_root_path, 
                filter_fn=lambda x: x['accent'] == 'German' if args.mode == 'train' else lambda x: x['accent'] != 'German',
                data_tsf=pad_trunc(max_ms=1000, sample_rate=48000),
                include_rate=False
            )
    elif args.dataset == 'speech-commands':
        label_dict = inverse_dict(SpeechCommandsDataset.label_dic)
        if args.mode == 'full':
            train_dataset = SpeechCommandsDataset(
                root_path=args.dataset_root_path, mode='train',
                include_rate=False, data_type='all',
                data_tfs=pad_trunc(max_ms=1000, sample_rate=16000)
            )
            test_dataset = SpeechCommandsDataset(
                root_path=args.dataset_root_path, mode='test',
                include_rate=False, data_type='all',
                data_tfs=pad_trunc(max_ms=1000, sample_rate=16000)
            )
            ignore_test_clip = True
        else:
            dataset = SpeechCommandsDataset(
                root_path=args.dataset_root_path, mode=args.mode, 
                include_rate=False, data_type='all',
                data_tfs=pad_trunc(max_ms=1000, sample_rate=16000),
            )
    elif args.dataset == 'speech-commands-numbers':
        label_dict = inverse_dict(SpeechCommandsDataset.numbers)
        if args.mode == 'full':
            train_dataset = SpeechCommandsDataset(
                root_path=args.dataset_root_path, mode='train',
                include_rate=False, data_type='numbers',
                data_tfs=pad_trunc(max_ms=1000, sample_rate=16000)
            )
            test_dataset = SpeechCommandsDataset(
                root_path=args.dataset_root_path, mode='test',
                include_rate=False, data_type='numbers',
                data_tfs=pad_trunc(max_ms=1000, sample_rate=16000)
            )
            ignore_test_clip = True
        else:
            dataset = SpeechCommandsDataset(
                root_path=args.dataset_root_path, mode=args.mode,
                include_rate=False, data_type='numbers',
                data_tfs=pad_trunc(max_ms=1000, sample_rate=16000)
            )
    elif args.dataset == 'cifar-10':
        label_dict = {
            0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 
            5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'
        }
        if args.mode == 'full':
            train_dataset = CIFAR10(
                root=args.dataset_root_path, download=True, train=True, transform=v_transforms.ToTensor()
            )
            test_dataset = CIFAR10(
                root=args.dataset_root_path, download=True, train=False, transform=v_transforms.ToTensor()
            )
            ignore_test_clip = False
        else:
            dataset = CIFAR10(
                root=args.dataset_root_path, download=True, train=True if args.mode == 'train' else False,
                transform=v_transforms.ToTensor()
            )
    else:
        raise Exception('No support')
    
    if args.mode == 'full':
        if args.rate < 1.0:
            print('using ClipDataset')
            train_dataset = ClipDataset(dataset=train_dataset, rate=args.rate)
            if not ignore_test_clip:
                test_dataset = ClipDataset(dataset=test_dataset, rate=args.rate)
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=False,
            drop_last=False, num_workers=args.num_workers
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
            drop_last=False, num_workers=args.num_workers
        )
        tsne_data = cal_tSNEs(
            loaders={'train': train_loader, 'test': test_loader},
            label_dict=label_dict,
            reduceable=False if args.no_reduce else True
        )
    else:
        if args.rate < 1.0:
            print('using ClipDataset')
            dataset = ClipDataset(dataset=dataset, rate=args.rate)
        data_loader = DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=False, 
            drop_last=False, num_workers=args.num_workers
        )

        tsne_data = cal_tSNE(data_loader=data_loader, label_dict=label_dict, reduceable=False if args.no_reduce else True)
    tsne_data.to_csv(os.path.join(output_full_root_path, args.output_file))