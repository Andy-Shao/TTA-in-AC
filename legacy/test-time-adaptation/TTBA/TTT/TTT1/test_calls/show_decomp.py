import argparse
import numpy as np

import torch

def plot_losses(cls_losses: np.ndarray, ssh_losses: list[np.ndarray], fname: str, use_agg=True) -> None:
    from lib.misc import normalize
    import matplotlib.pyplot as plt

    if use_agg: 
        plt.switch_backend('agg')

    colors = ['r', 'g', 'b', 'm']
    labels = range(4)
    cls_losses = normalize(cls_losses)
    for losses, color, label in zip(ssh_losses, colors, labels):
        losses = normalize(losses)
        plt.scatter(cls_losses, losses, label=str(label), color=color, s=4)
        plt.xlabel('classification loss')
        plt.ylabel('rotation loss')
        plt.savefig('%s_scatter_%d.pdf' %(fname, label))
        plt.close()

def show_decomp(cls_initial: float, cls_correct: np.ndarray, all_ssh_initial: list[float], all_ssh_correct: list[np.ndarray], fname: str, args: argparse.Namespace, use_agg=False) -> None:
    if use_agg:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lib.test_helpers import count_each, pair_buckets

    labels = range(4)
    for ssh_initial, ssh_correct, label in zip(all_ssh_initial, all_ssh_correct, labels):
        print('Direction %d error %.2f' %(label, ssh_initial*100))

        dtrue = count_each(pair_buckets(cls_correct, ssh_correct))
        torch.save(dtrue, '%s_dec_%d.pth' %(fname, label))
        print('Error decoposition:', *dtrue)

        if args.silent:
            continue
        drand = decomp_rand(cls_initial, ssh_initial, sum(dtrue))
        width = 0.25
        ind = np.arange(4)
        plt.bar(ind, 		drand, width, label='independent')
        plt.bar(ind+width, 	dtrue, width, label='observed')

        plt.ylabel('count for label %d' %(label))
        plt.xticks(ind + width/2, ('RR', 'RW', 'WR', 'WW'))
        plt.legend(loc='best')
        plt.savefig('%s_bar_%d.pdf' %(fname, label))
        plt.close()

def decomp_rand(clse: float, sshe: float, total: float) -> tuple[int, int, int, int]:
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument('--corruption', default='original')
    parser.add_argument('--outf', default='.')
    parser.add_argument('--silent', default=False, type=bool)
    args = parser.parse_args()

    rdict = torch.load(args.outf + '/%s_%d_inl.pth' %(args.corruption, args.level))
    fname = args.outf + '/%s_%d' %(args.corruption, args.level)

    if not args.silent:
        plot_losses(cls_losses=rdict['cls_losses'], ssh_losses=rdict['ssh_losses'], fname=fname, use_agg=True)
    show_decomp(cls_initial=rdict['cls_initial'], cls_correct=rdict['cls_correct'], all_ssh_initial=rdict['ssh_initial'], 
        all_ssh_correct=rdict['ssh_correct'], fname=fname, use_agg=True, args=args)