import argparse
from subprocess import call

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=5)
    parser.add_argument('--shared', type=str, default='layer2')
    parser.add_argument('--setting', type=str, default='slow')
    parser.add_argument('--name', type=str, default='gn_expand')
    parser.add_argument('--dataroot', type=str, default='./data/')
    parser.add_argument('--fix_type', type=type, default='none')
    parser.add_argument('--batch_size_main', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=32)
    parser.add_argument('--corruption', type=str)
    parser.add_argument('--silent', type=bool, default=False)
    parser.add_argument('--dset_size', type=int, default=0)

    args = parser.parse_args()
    level = args.level
    shared = args.shared
    setting = args.setting
    name = args.name
    dataroot = f'--dataroot {args.dataroot}'

    if level == 0:
        common_corruptions = ['cifar_new']
    else: 
        common_corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
	        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
	        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
        if level == 5:
            common_corruptions.append('original')
    
    fix_bn = args.fix_type == 'fix_bn'
    fix_str = '_fix_bn' if fix_bn else ''
    fix_tag = '--fix_bn' if fix_bn else ''

    fix_ssh = args.fix_type == 'fix_ssh'
    fix_str = '_fix_ssh' if fix_ssh else ''
    fix_tag = '--fix_ssh' if fix_ssh else ''
    gpnorm_tag = '--group_norm 8' if name[:2] == 'gn' else ''
    none_tag = '--none' if shared == 'none' else ''

    dset_size = args.dset_size
    if setting == 'fast':
        lr = 0.001
        niter = 1
        online_tag = ''
    elif setting == 'medium':
        lr = 0.001
        niter = 3
        online_tag = ''
    elif setting == 'slow':
        lr = 0.001
        niter = 10
        online_tag = ''
    elif setting == 'jump':
        lr = 0.01
        niter = 1
        online_tag = ''
    elif setting == 'online':
        lr = 0.001
        niter = 1
        online_tag = '--online'
        dset_size = 10000

    batch_size_main = args.batch_size_main
    batch_size_test = args.batch_size_test
    for corruption in common_corruptions:
        if args.corruption is not None and args.corruption != corruption:
            continue
        print(corruption, level)
        args_str = ' '.join([
            'python', 'test_calls/test_initial.py',
            dataroot,
            gpnorm_tag,
            none_tag,
            '--grad_corr',
            '--level 		%d' %(level),
            '--corruption	%s' %(corruption),
            '--shared 		%s' %(shared),
            '--batch_size	%d'	%(batch_size_main),
            '--resume 		results/cifar10_%s_%s/' %(shared, name),
            '--outf 		results/C10C_%s_%s_%s%s/' %(shared, setting, name, fix_str)
        ])
        call(args=args_str, shell=True)
        if shared == 'none':
            continue
        
        args_str = ' '.join([
            'python', 'test_calls/show_decomp.py',
			'--level 		%d' %(level),
			'--corruption	%s' %(corruption),
			'--outf 		results/C10C_%s_%s_%s%s/' %(shared, setting, name, fix_str),
            '--silent %s' % (str(args.silent))])
        call(args=args_str, shell=True)

        args_str = ' '.join([
            'python', 'test_calls/test_adapt.py',
			dataroot,
			gpnorm_tag,
			online_tag,
			fix_tag,
			'--level 		%d' %(level),
			'--corruption	%s' %(corruption),
			'--shared 		%s' %(shared),
			'--batch_size	%d'	%(batch_size_test),
			'--lr 			%f' %(lr),
			'--niter		%d' %(niter),
			'--resume 		results/cifar10_%s_%s/' %(shared, name),
			'--outf 		results/C10C_%s_%s_%s%s/' %(shared, setting, name, fix_str),
            '--dset_size    %d' %(dset_size)])
        call(args=args_str, shell=True)

        args_str = ' '.join([
            'python', 'test_calls/show_result.py',
			'--analyze_bin',
			'--analyze_ssh',
			'--level 		%d' %(level),
			'--corruption	%s' %(corruption),
			'--outf 		results/C10C_%s_%s_%s%s/' %(shared, setting, name, fix_str),
            '--dset_size    %d' %(dset_size)])
        call(args=args_str, shell=True)