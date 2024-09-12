import argparse


def get_args(env='python'):
    parser = argparse.ArgumentParser(description='Denoising')
    # Arguments
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--noise-type', default='Real', type=str, required=False,
                        choices=['Poisson-Gaussian'],
                        help='(default=%(default)s)')
    parser.add_argument('--loss-function', default='MSE_Affine', type=str, required=False,
                        choices=['MSE', 'N2V', 'MSE_Affine', 'Noise_est', 'EMSE_Affine'],
                        help='(default=%(default)s)')
    parser.add_argument('--model-type', default='final', type=str, required=False,
                        choices=['case1',
                                 'case2',
                                 'case3',
                                 'case4',
                                 'case5',
                                 'case6',
                                 'case7',
                                 'FBI_Net',
                                 'PGE_Net',
                                 'DBSN',
                                 'FC-AIDE'],
                        help='(default=%(default)s)')
    parser.add_argument('--data-type', default='RawRGB', type=str, required=False,
                        choices=['Grayscale',
                                 'RawRGB',
                                 'FMD',],
                        help='(default=%(default)s)')
    parser.add_argument('--data-name', default='SEM_semi', type=str, required=False,
                        choices=['SEM_semi'],
                        help='(default=%(default)s)')
    
    parser.add_argument('--nepochs', default=50, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch-size', default=4, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.001, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--drop-rate', default=0.5, type=float, help='(default=%(default)f)')
    parser.add_argument('--drop-epoch', default=10, type=int, help='(default=%(default)f)')
    parser.add_argument('--crop-size', default=120, type=int, help='(default=%(default)f)')
    
    parser.add_argument('--alpha', default=0.01, type=float, help='(default=%(default)f)')
    parser.add_argument('--beta', default=0.02, type=float, help='(default=%(default)f)')
    
    parser.add_argument('--num-layers', default=8, type=int, help='(default=%(default)f)')
    parser.add_argument('--num-filters', default=64, type=int, help='(default=%(default)f)')
    parser.add_argument('--mul', default=1, type=int, help='(default=%(default)f)')
    
    
    parser.add_argument('--unet-layer', default=3, type=int, help='(default=%(default)f)')
    parser.add_argument('--pge-weight-dir', default=None, type=str, help='(default=%(default)f)')
    
    parser.add_argument('--output-type', default='sigmoid', type=str, help='(default=%(default)f)')
    parser.add_argument('--sigmoid-value', default=0.1, type=float, help='(default=%(default)f)')
    
    parser.add_argument('--use-other-target', action='store_true', help='For SEM_semi image, use other noisy image as target. \
        In PGE-Net evaluation, it denotes specific f number (noisy level), not all F numbers')
    parser.add_argument('--x-f-num', default='F1', type=str, help='For SEM_semi image, set input of f-number 8,16,32,64',
                        choices=['F8','F16','F32','F64'])
    parser.add_argument('--y-f-num', default='F64', type=str, help='For SEM_semi image, set target of f-number 8,16,32,64',
                        choices=['F8','F16','F32','F64'])
    parser.add_argument('--integrate-all-set', action='store_true', help='For SEM_semi image, no matter what f-number is, integrate all set')
    parser.add_argument('--set-num', default=1, type=int, help='For SEM_semi image, need f-number 8,16,32,64',
                        choices=[1,2,3,4])
    parser.add_argument('--test', action='store_true', help='For SEM_semi image, train dataset to be test dataset(small size)')
    parser.add_argument('--log-off', action='store_true', help='log off')
    parser.add_argument('--save-whole-model', action='store_true', help='save whole model')
    parser.add_argument('--train-with-MSEAffine', action='store_true', help='For SEM_semi image, clean image is denoised image with MSE_AFFINE,not F64 image')
    if env == 'python':
        args=parser.parse_args() # args=parser.parse_args(args=[])
    else :
        args=parser.parse_args(args=[])
    return args




