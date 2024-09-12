import argparse
from random import choice


def get_args(env=None):
    parser = argparse.ArgumentParser(description='Denoising')
    # Arguments
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--noise-type', default='Real', type=str, required=False,
                        choices =['Poisson-Gaussian'],
                        help='(default=%(default)s)')
    parser.add_argument('--loss-function', default='EMSE_Affine', type=str, required=False,
                        choices=['MSE', 'N2V', 'MSE_Affine', 'Noise_est', 'EMSE_Affine'],
                        help='(default=%(default)s)')
    parser.add_argument('--lambda-val', default=0.005, type=float, help='(default=%(default)f)')
    parser.add_argument('--apply-RPM', action='store_true', help='apply random pixel mask on input image')
    parser.add_argument('--RPM-type', default='random', type=str, required=False,
                        choices=['random', 'grid','mixed'],)
    parser.add_argument('--RPM-grid-size', default=0, type=int, help='(default=%(default)f)')
    parser.add_argument('--RPM-masking-value',default=0, type=float, help='(default=%(default)f)')
    parser.add_argument('--RPM-p', default=0.5, type=float, help='(default=%(default)f)')
    parser.add_argument('--vst-version', default='MSE', type=str, required=False,
                        choices=['MSE', 'MAE'])
    parser.add_argument('--model-type', default='final', type=str, required=False,
                        choices=['FBI_Net',
                                 'PGE_Net',
                                 'DBSN',
                                 'UNet',
                                 'NAFNet_light',
                                 'NAFNet',
                                 'DBSN',
                                 'FC-AIDE',
                                 'RED30',
                                 'N2N_UNet',
                                 'N2N_UNet_BN',
                                 'N2N_UNet_BN_input',
                                 'FBI_Net_BN',
                                 'FBI_Net_dropout',
                                 'N2N_UNet_dropout',
                                 'FBI_Net_tiny'],
                        help='(default=%(default)s)')
    parser.add_argument('--non-blind', action='store_true', help='(default=%(default)s), set non-blind in FBI-Net')
    parser.add_argument('--data-type', default=None, type=str, required=False,
                        choices=['Grayscale',
                                 'RawRGB'],
                        help='(default=%(default)s)')
    parser.add_argument('--data-name', default=None, type=str, required=False,
                        choices=['BSD',
                                 'fivek',
                                 'SIDD',
                                 'DND',
                                 'SEM_semi'],
                        help='(default=%(default)s)')
    
    parser.add_argument('--nepochs', default=10, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch-size', default=4, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.001, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--weight-decay', default=0, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--drop-rate', default=0.5, type=float, help='(default=%(default)f)')
    parser.add_argument('--drop-epoch', default=10, type=int, help='(default=%(default)f)')
    parser.add_argument('--crop-size', default=120, type=int, help='(default=%(default)f)')
    
    parser.add_argument('--alpha', default=0.01, type=float, help='(default=%(default)f)')
    parser.add_argument('--beta', default=0.02, type=float, help='(default=%(default)f)')
    parser.add_argument('--test-alpha', default=0, type=float, help='(default=%(default)f)')
    parser.add_argument('--test-beta', default=0, type=float, help='(default=%(default)f)')

    # parser.add_argument('--train-set', type=str, nargs="+", help='To specify train set')
    parser.add_argument('--test-wholedataset', action='store_true', help='To test on wholedataset')
    parser.add_argument('--wholedataset-version', default='v1', type=str, 
                            choices = ["None",'v1', 'v2'],
                            help='Select wholedataset version \
                            v1 : SET01~SET04, v2 : SET05~SET10(default=%(default)f)')
    parser.add_argument('--test-set', type=str, nargs="+", help='To specify test set')
    parser.add_argument('--num-layers', default=8, type=int, help='(default=%(default)f)')
    parser.add_argument('--num-filters', default=64, type=int, help='(default=%(default)f)')
    parser.add_argument('--mul', default=1, type=int, help='(default=%(default)f)')
    
    
    parser.add_argument('--unet-layer', default=3, type=int, help='(default=%(default)f)')
    parser.add_argument('--pge-weight-dir', default=None, type=str, help='(default=%(default)f)')
    
    parser.add_argument('--output-type', default='sigmoid', type=str, help='(default=%(default)f)')
    parser.add_argument('--sigmoid-value', default=0.1, type=float, help='(default=%(default)f)')
    
    parser.add_argument('--use-other-target', action='store_true', help='For SEM_semi image, use other noisy image as target. \
        In PGE-Net evaluation, it denotes specific f number (noisy level), not all F numbers')
    parser.add_argument('--x-f-num', default='F#', type=str, help='For SEM_semi image, set input of f-number 8,16,32,64',
                        choices=['F#', 'F1','F2','F4','F8','F8', 'F01','F02','F04','F08','F16','F32','F64'])
    parser.add_argument('--y-f-num', default='F64', type=str, help='For SEM_semi image, set target of f-number 8,16,32,64',
                        choices=['F1','F2','F4','F8','F01','F02','F04','F08','F16','F32','F64','F#'])
    parser.add_argument('--clean-f-num', default='F64', type=str, help='For SEM_semi image, set target of f-number 8,16,32,64',
                        choices=['F32','F64'])
    parser.add_argument('--integrate-all-set', action='store_true', help='For SEM_semi image, no matter what f-number is, integrate all set')
    parser.add_argument('--individual-noisy-input', action='store_true', help='For SEM_semi image, no matter what f-number is, integrate all set')
    parser.add_argument('--dataset-type', default='train', type=str, help='For SEM_semi image, train dataset or test dataset',
                        choices=['train','test','val'])
    parser.add_argument('--batch-test', action='store_true', help='batch test')
    parser.add_argument('--test', action='store_true', help='For SEM_semi image, train dataset to be test dataset(small size)')
    parser.add_argument('--log-off',action='store_true', help='logger (neptune) off')
    parser.add_argument('--save-whole-model',action='store_true', help='save whole model')
    parser.add_argument('--speed-test',action='store_true', help='for speed test')
    parser.add_argument('--apply_median_filter',action='store_true', help='apply median_filter instead of FBI-Net')
    parser.add_argument('--apply_median_filter_target',action='store_true', help='apply median_filter to target image')
    parser.add_argument('--train-with-MSEAffine', action='store_true', help='For SEM_semi image, clean image is denoised image with MSE_AFFINE,not F64 image')
    parser.add_argument('--with-originalPGparam', action='store_true', help='For noise estimation, not using PGE-Net, use original PG param(oracle)')
    parser.add_argument('--load-ckpt', default=None, type=str, help='(default=%(default)f) If you want to start from points you trained, use this')
    parser.add_argument('--mixed-target', action='store_true', help='For SEM_semi image, mixed target image for F01')
    parser.add_argument('--y-f-num-type',default='v1', type=str, help='default : v1 (cover whole f over F01), v2 : F02,F04,F08')
    parser.add_argument('--average-mode', default='mean', type=str, help='(default=%(default)f)', choices=['mean', 'median'])
    parser.add_argument('--testidx', default=7, type=int, help='(default=%(default)f)')
    parser.add_argument('--eval-off', action='store_true', help='(default=%(default)f)')
    parser.add_argument('--input-feature-normalize', action='store_true', help='(default=%(default)f)')
    parser.add_argument('--input-mean-std-normalize', action='store_true', help='(default=%(default)f)')
    parser.add_argument('--input-dropout', action='store_true', help='(default=%(default)f)')
    parser.add_argument('--dropout-rate',default=0.6, type=float, help='(default=%(default)f)')
    parser.add_argument('--rescale-after-RPM', action='store_true', help='(default=%(default)f)')
    parser.add_argument('--rescale-on-eval', action='store_true', help='(default=%(default)f)')
    parser.add_argument('--dropout-RPM-mode', action='store_true', help='(default=%(default)f)')
    parser.add_argument('--dropout-type',default=None)
    parser.add_argument('--train-time-RPM', action='store_true', help='(default=%(default)f)')
    parser.add_argument('--apply-RPM-before-model', action='store_true', help='(default=%(default)f)')
    parser.add_argument('--add-noise-term', action='store_true', help='(default=%(default)f)')
    if env == None:
        args=parser.parse_args()
    else :
        args=parser.parse_args([])
    
    return args





