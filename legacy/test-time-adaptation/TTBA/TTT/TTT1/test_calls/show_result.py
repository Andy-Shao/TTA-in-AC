import argparse
import numpy as np

import torch

from lib.misc import mean, print_color
from lib.test_helpers import pair_buckets

printed_head = 'show_result:'

def analyze(idx_tbd, idx_all, err):
    new_tbd = np.logical_and(idx_all, idx_tbd).sum()
    new_per = new_tbd.sum() / idx_tbd.sum()
    if err:
        print_color('RED', 		printed_head+'%d\t%d\t%.2f' %(idx_tbd.sum(), idx_tbd.sum() - new_tbd.sum(), (1-new_per)*100))
    else:
        print_color('GREEN',	printed_head+'%d\t%d\t%.2f' %(idx_tbd.sum(), new_tbd.sum(), new_per*100))

def analyze_all(adapted: np.ndarray, all_initial: tuple):
    errs = [True, True, False, False]
    for err, initial in zip(errs, all_initial):
        analyze(initial, adapted, err)

def show_result(adapted: float, initial: float):
    print(printed_head+'Error (%)')
    print_color('RED', 		printed_head+'initial error: %.1f' %(initial*100))
    print_color('YELLOW', 	printed_head + 'adapted error: %.1f' %(adapted*100))
    print_color('GREEN',	printed_head + 'initial - adapted error: %.1f' %((initial - adapted)*100))

def get_err_adapted(new_correct: np.ndarray, old_correct: np.ndarray, ssh_confide: np.ndarray, threshold=1) -> float:
    adapted = new_correct[ssh_confide < threshold]
    noadptd	= old_correct[ssh_confide >= threshold]
    return 1 - (sum(adapted) + sum(noadptd)) / (len(adapted) + len(noadptd))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument('--corruption', default='original')
    parser.add_argument('--outf', default='.')
    parser.add_argument('--threshold', default=1, type=float)
    parser.add_argument('--dset_size', default=0, type=int)
    parser.add_argument('--analyze_bin', action='store_true')
    parser.add_argument('--analyze_ssh', action='store_true')
    parser.add_argument('--save_oh', action='store_true')
	
    args = parser.parse_args()
    args.threshold += 0.001		# to correct for numeric errors
    rdict_ada = torch.load(args.outf + '/%s_%d_ada.pth' %(args.corruption, args.level))
    rdict_inl = torch.load(args.outf + '/%s_%d_inl.pth' %(args.corruption, args.level))

    ssh_confide = rdict_ada['ssh_confide']
    new_correct = rdict_ada['cls_correct']
    old_correct = rdict_inl['cls_correct']

    if args.dset_size == 0:
        args.dset_size = len(old_correct)

    old_correct = old_correct[:args.dset_size]
    err_adapted = get_err_adapted(new_correct, old_correct, ssh_confide, threshold=args.threshold)
    show_result(err_adapted, 1-mean(old_correct))

    if args.analyze_bin:
        print(printed_head + 'Bin analysis')
        for label, ssh_correct in enumerate(rdict_inl['ssh_correct']):
            ssh_correct = ssh_correct[:args.dset_size]
            dvecs = pair_buckets(old_correct, ssh_correct)
            print('Direction %d' %(label))
            analyze_all(rdict_ada['cls_correct'], dvecs)

    if args.analyze_ssh:
        print(printed_head + 'Self-supervised analysis')
        for label, ssh_correct in enumerate(rdict_inl['ssh_correct']):
            ssh_correct = ssh_correct[:args.dset_size]
            trerror = 1 - mean([correct[label].item() for correct in rdict_ada['trerror']])
            print(printed_head + 'Direction %d' %(label))
            print(printed_head + ' Old error (%%): %.2f' %((1-mean(ssh_correct)) * 100))
            print(printed_head + ' New error (%%): %.2f' %(trerror * 100))
    
    if args.save_oh:
        torch.save((old_correct, new_correct), args.outf + '/one_hot_saved.pth')