import argparse

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch

from lib.misc import my_makedir
from lib.test_helpers import build_model, test, plot_epochs
from lib.prepare_dataset import prepare_test_data, prepare_train_data
from lib.rotation import rotate_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--dataroot', default='./data/')
    parser.add_argument('--shared', default=None)
    ########################################################################
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--width', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--group_norm', default=0, type=int)
    ########################################################################
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--nepoch', default=75, type=int)
    parser.add_argument('--milestone_1', default=50, type=int)
    parser.add_argument('--milestone_2', default=65, type=int)
    parser.add_argument('--rotation_type', default='rand')
    ########################################################################
    parser.add_argument('--outf', default='.')

    args = parser.parse_args()

    my_makedir(args.outf)
    cudnn.benchmark = True
    net, ext, head, ssh = build_model(args=args)
    _, test_loader = prepare_test_data(args=args)
    _, train_loader = prepare_train_data(args=args)

    parameters = list(net.parameters()) + list(head.parameters())
    optimizer = optim.SGD(params=parameters, lr=args.lr, momentum=.9, weight_decay=5e-4)
    """
    # Assuming optimizer uses lr = 0.05 for all groups
    # lr = 0.05     if epoch < 30
    # lr = 0.005    if 30 <= epoch < 80
    # lr = 0.0005   if epoch >= 80
    scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    """
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[args.milestone_1, args.milestone_2], gamma=.1, last_epoch=-1)
    criterion = nn.CrossEntropyLoss().cuda()

    all_err_cls = []
    all_err_ssh = []
    print('Running...')
    print('Error (%)\t\ttest\t\tself-supervised')
    for epoch in range(1, args.nepoch+1):
        net.train()
        ssh.train()

        for batch_idx, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            features_cls, labels_cls = features.cuda(), labels.cuda()
            outputs_cls = net(features_cls)
            loss = criterion(outputs_cls, labels_cls)

            if args.shared is not None:
                features_ssh, labels_ssh = rotate_batch(features, args.rotation_type)
                features_ssh, labels_ssh = features_ssh.cuda(), labels_ssh.cuda()
                outputs_ssh = ssh(features_ssh)
                loss_ssh = criterion(outputs_ssh, labels_ssh)
                loss += loss_ssh
            
            loss.backward()
            optimizer.step()
        
        err_cls = test(dataloader=test_loader, model=net)[0]
        err_ssh = 0 if args.shared else test(dataloader=test_loader, model=ssh, sslabel='expand')[0]
        all_err_cls.append(err_cls)
        all_err_ssh.append(err_ssh)
        scheduler.step()

        print(('Epoch %d/%d:' %(epoch, args.nepoch)).ljust(24) +
                    '%.2f\t\t%.2f' %(err_cls*100, err_ssh*100))
        
    torch.save((all_err_cls, all_err_ssh), args.outf + '/loss.pth')
    plot_epochs(all_err_cls=all_err_cls, all_err_ssh=all_err_ssh, fname=args.outf + '/loss.pdf')
    
    state = {
        'err_cls': err_cls, 'err_ssh': err_ssh,
        'net': net.state_dict(), 'head': head.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, args.outf + '/ckpt.pth')