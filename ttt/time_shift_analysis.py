import argparse
import os
import pandas as pd
from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.optim as optim

from lib.toolkit import print_argparse
from ttt.lib.test_helpers import build_mnist_model, time_shift_inference as inference
from ttt.lib.prepare_dataset import prepare_test_data, test_transforms, TimeShiftOps, train_transforms
from ttt.lib.time_shift_rotation import rotate_batch

def test_one(feature: torch.Tensor, model: nn.Module, data_transf: nn.Module) -> tuple[int, int]:
    model.eval()
    audios, labels = rotate_batch(batch=torch.unsqueeze(feature, dim=0), label='expand', data_transforms=data_transf)
    audios, labels = audios.to(device=args.device), labels.to(device=args.device)
    with torch.no_grad():
        outputs = model(audios)
        _, preds = torch.max(outputs, dim=1)
    correct_num = (preds == labels).sum().cpu().item()
    return correct_num, labels.shape[0]

def adapt_one(feature: torch.Tensor, ssh: nn.Module, ext: nn.Module, args: argparse.Namespace, 
              criterion: nn.Module, data_transf: nn.Module, optimizer: optim.Optimizer) -> None:
    ssh.eval()
    ext.train()
    for it in range(args.niter):
        features = torch.unsqueeze(feature, dim=0).repeat(args.batch_size, 1, 1)
        audios, labels = rotate_batch(batch=features, label='rand', data_transforms=data_transf)
        audios, labels = audios.to(args.device), labels.to(args.device)
        optimizer.zero_grad()
        outputs = ssh(audios)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def measure_one(model: nn.Module, audio: torch.Tensor, label: int) -> tuple[int, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        output = model(audio)
        _, pred = torch.max(output, dim=1)
        confidence = nn.functional.softmax(output, dim=1).squeeze()[label].item()
    correctness = 1 if pred.item() == label else 0
    return correctness, confidence

def load_model(args: argparse.Namespace, mode:str) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    assert mode in ['origin', 'adapted']
    net, ext, head, ssh = build_mnist_model(args)
    if mode == 'origin':
        stat = torch.load(args.origin_model_weight_file_path)
    elif mode == 'adapted':
        stat = torch.load(args.adapted_model_weight_file_path)
    net.load_state_dict(stat['net'])
    head.load_state_dict(stat['head'])
    return net, ext, head, ssh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    parser.add_argument('--origin_model_weight_file_path', type=str)
    parser.add_argument('--adapted_model_weight_file_path', type=str)
    parser.add_argument('--output_path', type=str, default='./result')
    parser.add_argument('--dataset_root_path', type=str)
    parser.add_argument('--rotation_type', default='rand')
    parser.add_argument('--group_norm', default=0, type=int)
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--severity_level', default=.0025, type=float)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--shared', type=str, default='layer2')
    ########################################################################
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--niter', default=1, type=int)
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--threshold', default=1, type=float)
    parser.add_argument('--shift_limit', default=.25, type=float)

    args = parser.parse_args()
    args.output_full_path = os.path.join(args.output_path, args.dataset, 'ttt', 'analysis')
    try:
        os.makedirs(args.output_full_path)
    except:
        pass

    accu_record = pd.DataFrame(columns=['dataset', 'algorithm', 'tta-operation', 'corruption', 'accuracy', 'error', 'severity level'])
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'audio-mnist':
        args.class_num = 10
        args.sample_rate = 48000
        args.n_mels = 64
        args.final_full_line_in = 384
        args.hop_length = 505
        args.ssh_class_num = 3
        net, ext, head, ssh = load_model(args, mode='origin')
    else:
        raise Exception('No support')
    print_argparse(args)
    # Finish args prepare
    
    print('Origin test')
    args.corruption = 'original'
    test_dataset, test_loader = prepare_test_data(args=args)
    test_transf = test_transforms(args)
    # original_test_accu = inference(model=net, loader=test_loader, test_transf=test_transf, device=args.device)
    # accu_record.loc[len(accu_record)] = [args.dataset, 'RestNet', 'N/A', 'N/A', original_test_accu, 100. - original_test_accu, 0.]
    # print(f'original data size: {len(test_dataset)}, original accuracy: {original_test_accu:.2f}%')

    print('Corruption test')
    args.corruption = 'gaussian_noise'
    corrupted_test_transf = test_transforms(args)
    corrupted_test_accu = inference(model=net, loader=test_loader, test_transf=corrupted_test_transf, device=args.device)
    accu_record.loc[len(accu_record)] = [args.dataset, 'RestNet', 'N/A', args.corruption, corrupted_test_accu, 100. - corrupted_test_accu, args.severity_level]
    print(f'corrupted data size: {len(test_dataset)}, corrupted accuracy: {corrupted_test_accu:.2f}%')

    print('Online ttt adaptation')
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    args.threshold += 0.001		# to correct for numeric errors
    args.corruption = 'gaussian_noise'
    args.batch_size = args.batch_size // 3
    train_transfs = train_transforms(args)
    if args.dataset == 'audio-mnist':
        net, ext, head, ssh = load_model(args, mode='origin')
    else:
        raise Exception('No support')
    criterion_ssh = nn.CrossEntropyLoss().to(device=args.device)
    optimizer_ssh = optim.SGD(params=ssh.parameters(), lr=args.lr)
    ttl_corr = 0
    for feature, label in tqdm(test_dataset):
        input = corrupted_test_transf[TimeShiftOps.ORIGIN].tran_one(feature)
        input = input.to(args.device)
        _, confidence = measure_one(model=ssh, audio=input, label=0)
        if confidence < args.threshold:
            adapt_one(feature=feature, ssh=ssh, ext=ext, args=args, criterion=criterion_ssh, data_transf=train_transfs, optimizer=optimizer_ssh)
        correctness, confidence = measure_one(model=net, audio=input, label=label)
        ttl_corr += correctness
    ttl_accu = ttl_corr / len(test_dataset) * 100.
    print(f'TTT adaptation data size: {len(test_dataset)}, accuracy: {ttl_accu}%')

    # accu_record.to_csv(os.path.join(args.output_full_path, 'accuracy_record.csv'))