import argparse
import os

from torch.utils.data import DataLoader

from lib.datasets import FilterAudioMNIST
from lib.scDataset import SpeechCommandsDataset
from lib.wavUtils import pad_trunc
from dataset_analysis.lib.tSEN_utils import cal_tSNE, inverse_dict
from lib.toolkit import print_argparse

if __name__ == '__main__':
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--dataset', type=str, default=['audio-mnist', 'speech-commands', 'speech-commands-numbers', 'speech-commands-random'])
    arg_parse.add_argument('--dataset_root_path', type=str)
    arg_parse.add_argument('--num_workers', type=int, default=16)
    arg_parse.add_argument('--output_root_path', type=str, default='./result')
    arg_parse.add_argument('--batch_size', type=int, default=256)
    arg_parse.add_argument('--output_file', type=str)
    arg_parse.add_argument('--mode', type=str, default=['train', 'test'])

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
            0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 
            5:'5', 6:'6', 7:'7', 8:'8', 9:'9'
        }
        dataset = FilterAudioMNIST(
            root_path=args.dataset_root_path, 
            filter_fn=lambda x: x['accent'] == 'German' if args.mode == 'train' else lambda x: x['accent'] != 'German',
            data_tsf=pad_trunc(max_ms=1000, sample_rate=48000),
            include_rate=False
        )
    elif args.dataset == 'speech-commands':
        label_dict = inverse_dict(SpeechCommandsDataset.label_dic)
        dataset = SpeechCommandsDataset(
            root_path=args.dataset_root_path, mode=args.mode, 
            include_rate=False, data_type='all',
            data_tfs=pad_trunc(max_ms=1000, sample_rate=16000),
        )
    else:
        raise Exception('No support')
    
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=False, 
        drop_last=False, num_workers=args.num_workers
    )

    tsne_data = cal_tSNE(data_loader=data_loader, label_dict=label_dict)
    tsne_data.to_csv(os.path.join(output_full_root_path, args.output_file))