import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from lib.misc import my_makedir, mean
from lib.test_helpers import build_model
from lib.prepare_dataset import prepare_test_data, test_transforms, train_transforms
from lib.rotation import rotate_batch, rotate_batch_with_labels

def trerr_single(model: nn.Module, image: Image) -> torch.Tensor:
    model.eval()
    labels = torch.LongTensor([0, 1, 2, 3])
    inputs = torch.stack([test_transforms(image) for _ in range(4)])
    inputs = rotate_batch_with_labels(batch=inputs, labels=labels)
    inputs, labels = inputs.cuda(), labels.cuda()
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim=1)
    return predicted.eq(other=labels).cpu()

def adapt_single(image: Image, ssh: nn.Module, ext: nn.Module, args: argparse.Namespace, criterion: nn.Module):
    if args.fix_bn:
        ssh.eval()
    elif args.fix_ssh:
        ssh.eval()
        ext.eval()
    else:
        ssh.train()
    for it in range(args.niter):
        inputs = [train_transforms(image) for _ in range(args.batch_size)]
        inputs = torch.stack(inputs)
        inputs_ssh, labels_ssh = rotate_batch(batch=inputs, label='rand')
        inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
        optimizer_ssh.zero_grad()
        outputs_ssh = ssh(inputs_ssh)
        loss_ssh = criterion(outputs_ssh, labels_ssh)
        loss_ssh.backward()
        optimizer_ssh.step()

def test_single(model: nn.Module, image: Image, label: torch.Tensor) -> tuple[int, float]:
    model.eval()
    inputs = torch.unsqueeze(test_transforms(img=image), dim=0)
    with torch.no_grad():
        outputs = model(inputs.cuda())
        _, predicted = torch.max(outputs, dim=1)
        confidence = nn.functional.softmax(outputs, dim=1).squeeze()[label].item()
    correctness = 1 if predicted.item() == label else 0
    return correctness, confidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument('--corruption', default='original')
    parser.add_argument('--dataroot', default='/data/yusun/datasets/')
    parser.add_argument('--shared', default=None)
    ########################################################################
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--group_norm', default=0, type=int)
    parser.add_argument('--fix_bn', action='store_true')
    parser.add_argument('--fix_ssh', action='store_true')
    ########################################################################
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--niter', default=1, type=int)
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--threshold', default=1, type=float)
    parser.add_argument('--dset_size', default=0, type=int)
    ########################################################################
    parser.add_argument('--outf', default='.')
    parser.add_argument('--resume', default=None)

    args = parser.parse_args()
    args.threshold += 0.001		# to correct for numeric errors
    my_makedir(args.outf)
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    net, ext, head, ssh = build_model(args=args)
    test_set, test_loader = prepare_test_data(args=args)
    printed_head = 'test_adapt:'

    print('%s Resuming from %s...' %(printed_head, args.resume))
    ckpt = torch.load(args.resume + '/ckpt.pth')
    if args.online:
        net.load_state_dict(ckpt['net'])
        head.load_state_dict(ckpt['head'])
    
    criterion_ssh = nn.CrossEntropyLoss().cuda()
    if args.fix_ssh:
        optimizer_ssh = optim.SGD(params=ext.parameters(), lr=args.lr)
    else:
        optimizer_ssh = optim.SGD(params=ssh.parameters(), lr=args.lr)

    print(f'{printed_head} Running...')
    correct = []
    ssh_conf = []
    train_error = []
    if args.dset_size == 0:
        args.dset_size = len(test_set)
    
    for i in tqdm(range(1, args.dset_size+1)):
        if not args.online:
            net.load_state_dict(ckpt['net'])
            head.load_state_dict(ckpt['head'])

        _, label = test_set[i - 1]
        image = Image.fromarray(test_set.data[i - 1])

        ssh_conf.append(test_single(model=ssh, image=image, label=0)[1])
        if ssh_conf[-1] < args.threshold:
            adapt_single(image=image, ssh=ssh, ext=ext, args=args, criterion=criterion_ssh)
        correct.append(test_single(model=net, image=image, label=label)[0])
        train_error.append(trerr_single(model=ssh, image=image))
    
    rdict = {'cls_correct': np.asarray(correct), 'ssh_confide': np.asarray(ssh_conf), 
		'cls_adapted':1-mean(correct), 'trerror': train_error}
    torch.save(rdict, args.outf + '/%s_%d_ada.pth' %(args.corruption, args.level))
