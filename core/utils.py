import sys,os
import random
import time
import datetime
from typing import List
import numpy as np
import scipy.io as sio
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvF
import h5py
import random
import torch
from torch.autograd import Variable
import math
#from skimage import measure # measure.compare_ssim is deprecated after 0.19
import sys
import matplotlib.pyplot as plt
from copy import deepcopy
from skimage.metrics import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

class TrdataLoader():

    def __init__(self, _tr_data_dir=None, _args = None):
        # print(sys.path)
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.tr_data_dir = _tr_data_dir
        self.args = _args
        self.data = h5py.File(self.tr_data_dir, "r")
        self.noisy_arr, self.clean_arr = None, None
        
        self.noisy_arr = self.data[f"{self.args.x_f_num}_1"]
        self.clean_arr = self.data[f"{self.args.y_f_num}_2"]
        
        print("noisy_arr : ",self.noisy_arr.shape, f"pixel value range from {self.noisy_arr[-1].min()} ~ {self.noisy_arr[-1].max()}")
        print("clean_arr : ",self.clean_arr.shape, f"pixel value range from {self.clean_arr[-1].min()} ~ {self.clean_arr[-1].max()}")

        self.num_data = self.clean_arr.shape[0]
        print ('num of training atches : ', self.num_data)
        
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, index):
        
        # random crop
        

        if self.args.noise_type == 'Gaussian' or self.args.noise_type == 'Poisson-Gaussian':

            clean_img = self.clean_arr[index,:,:]
            noisy_img = self.noisy_arr[index,:,:]
            noisy_img = (noisy_img - noisy_img.min()) / (noisy_img.max() - noisy_img.min())
            clean_img = (clean_img - clean_img.min()) / (clean_img.max() - clean_img.min())
            if self.args.data_type == 'Grayscale':
                rand = random.randrange(1,10000)
                clean_patch,noisy_patch = None,None
                if clean_img.shape[0] <= self.args.crop_size and clean_img.shape[1] <= self.args.crop_size:
                    clean_patch = clean_img
                    noisy_patch = noisy_img
                else :
                    clean_patch = Image.extract_patches_2d(image = clean_img ,patch_size = (self.args.crop_size, self.args.crop_size), 
                                             max_patches = 1, random_state = rand)
                    noisy_patch = Image.extract_patches_2d(image = noisy_img ,patch_size = (self.args.crop_size, self.args.crop_size), 
                                             max_patches = 1, random_state = rand)
                
                    # Random horizontal flipping
                if random.random() > 0.5:
                    clean_patch = np.fliplr(clean_patch)
                    noisy_patch = np.fliplr(noisy_patch)

                # Random vertical flipping
                if random.random() > 0.5:
                    clean_patch = np.flipud(clean_patch)
                    noisy_patch = np.flipud(noisy_patch)
                if self.args.apply_RPM is True:
                    noisy_patch = apply_RPM(noisy_patch=noisy_patch,RPM_p=self.args.RPM_p,
                                            RPM_type=self.args.RPM_type,RPM_grid_size=self.args.RPM_grid_size,
                                            RPM_masking_value=self.args.RPM_masking_value)
                    if self.args.rescale_after_RPM:
                        noisy_patch = noisy_patch * (1/(1-self.args.RPM_p))
                # need to expand_dims since grayscale has no channel dimension
                clean_patch, noisy_patch = np.expand_dims(clean_patch,axis=0), np.expand_dims(noisy_patch,axis=0)
            else:

                rand_x = random.randrange(0, (clean_img.shape[0] - self.args.crop_size -1) // 2)
                rand_y = random.randrange(0, (clean_img.shape[1] - self.args.crop_size -1) // 2)

                clean_patch = clean_img[rand_x*2 : rand_x*2 + self.args.crop_size, rand_y*2 : rand_y*2 + self.args.crop_size].reshape(1, self.args.crop_size, self.args.crop_size)
                noisy_patch = noisy_img[rand_x*2 : rand_x*2 + self.args.crop_size, rand_y*2 : rand_y*2 + self.args.crop_size].reshape(1, self.args.crop_size, self.args.crop_size)

            

            if self.args.loss_function == 'MSE' :
            
                source = torch.from_numpy(noisy_patch.copy())
                target = torch.from_numpy(clean_patch.copy())
            
            elif self.args.loss_function == 'MSE_Affine' or self.args.loss_function == 'N2V' or self.args.loss_function == 'Noise_est' or self.args.loss_function == 'EMSE_Affine':
                
                source = torch.from_numpy(noisy_patch.copy())
                target = torch.from_numpy(clean_patch.copy())
                
                target = torch.cat([source,target], dim = 0) # (512,256) -> (2,256,256)
            return source, target

        else: ## real data

            return source, target

class TedataLoader():

    def __init__(self,_te_data_dir=None, args = None):
        """
        te_data_dir : path to test data or single image
        """

        self.te_data_dir = _te_data_dir
        self.args = args
        if 'SIDD' in self.te_data_dir or 'DND' in self.te_data_dir or 'CF' in self.te_data_dir or 'TP' in self.te_data_dir:
            self.data = sio.loadmat(self.te_data_dir)
        elif self.te_data_dir.endswith('.hdf5'):
            self.data = h5py.File(self.te_data_dir, "r")
        elif self.te_data_dir.endswith('.npy'):
            self.data = np.load(self.te_data_dir, allow_pickle=True)
            self.data = self.data.item()
        else :
            raise ValueError('te_data_dir has to be .mat or .h5f5 or .npy')
        self.noisy_arr, self.clean_arr = None, None
        # [3086 * i, 3086 * (i+1)]
        self.noisy_arr = self.data[f"{self.args.x_f_num}_1"]
        self.clean_arr = self.data[f"{self.args.clean_f_num}_2"]
        self.num_data = self.noisy_arr.shape[0]
        
        print ('num of test images : ', self.num_data)

    def __len__(self):
        return self.num_data
    
    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""
        
        source = self.noisy_arr[index,:,:]
        target = self.clean_arr[index,:,:]
        
        source = (source - source.min()) / (source.max() - source.min())
        target = (target - target.min()) / (target.max() - target.min())
        
        if 'SIDD' in self.te_data_dir or 'CF' in self.te_data_dir or 'TP' in self.te_data_dir:
            source = source / 255.
            target = target / 255.

        source = torch.from_numpy(source.reshape(1,source.shape[0],source.shape[1])).float().cuda()
        target = torch.from_numpy(target.reshape(1,target.shape[0],target.shape[1])).float().cuda()
        # print("get_item : ",self.args.loss_function[:10] )
        if self.args.loss_function == 'MSE_Affine' or self.args.loss_function == 'N2V':
            target = torch.cat([source,target], dim = 0)

        return source, target

class TedataLoader_FullImage():

    def __init__(self,_te_data_dir=None, args = None):
        """
        te_data_dir : path to test data or single image
        """

        self.te_data_dir = _te_data_dir
        self.args = args
        self.data = load_img_dict( f"{self.args.x_f_num}_1",f"{self.args.clean_f_num}_2",
                clean_f_num=self.args.clean_f_num,debug=False)
        self.noisy_arr, self.clean_arr = None, None
        # [3086 * i, 3086 * (i+1)]
        mask = np.zeros(len(self.data[f"{self.args.x_f_num}_1"])).astype(bool)
        try:
            img_idx_info = np.load("./dataset_index_info.npy")
        except:
            raise ValueError("dataset_index_info.npy is not found, which is generated from 4_make_patch.py within data_preparation")
        for idx, img_idx in enumerate(img_idx_info):
            if img_idx == 'test':
                mask[idx] = True
            else:
                mask[idx] = False
            
        self.noisy_arr = np.array(self.data[f"{self.args.x_f_num}_1"])[mask] # num,ch(=1),h,w
        self.clean_arr = np.array(self.data[f"{self.args.clean_f_num}_2"])[mask]
        assert self.noisy_arr.shape == self.clean_arr.shape
        self.num_data = self.noisy_arr.shape[0]
        
        self.proper_height = 256*(self.noisy_arr[0].shape[1]//256) # 1280
        self.proper_width = 256*(self.noisy_arr[0].shape[2]//256) # 2816
        # print(self.noisy_arr[0][:,:self.proper_height,:self.proper_width].shape)
        print ('num of test images : ', self.num_data)#, self.noisy_arr.shape, self.clean_arr.shape)

    def __len__(self):
        return self.num_data
    
    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""
        
        source = self.noisy_arr[index][:,:self.proper_height,:self.proper_width]
        target = self.clean_arr[index][:,:self.proper_height,:self.proper_width]
        
        source = (source - source.min()) / (source.max() - source.min())
        target = (target - target.min()) / (target.max() - target.min())

        source = torch.from_numpy(source.reshape(1,source.shape[1],source.shape[2])).float().cuda()
        target = torch.from_numpy(target.reshape(1,target.shape[1],target.shape[2])).float().cuda()
        # print("get_item : ",self.args.loss_function[:10] )
        if self.args.loss_function == 'MSE_Affine' or self.args.loss_function == 'N2V':
            target = torch.cat([source,target], dim = 0)

        return source, target
def get_PSNR(X, X_hat):
    if type(X) == torch.Tensor:
        X = X.cpu().numpy()
    if type(X_hat) == torch.Tensor:
        X_hat = X_hat.cpu().numpy()

    mse = np.mean((X-X_hat)**2)
    test_PSNR = 10 * math.log10(1/mse)
    
    return test_PSNR

def get_SSIM(X, X_hat):
    if type(X) == torch.Tensor:
        X = X.cpu().numpy()
    if type(X_hat) == torch.Tensor:
        X_hat = X_hat.cpu().numpy()
    
    ch_axis = 0
    #test_SSIM = measure.compare_ssim(np.transpose(X, (1,2,0)), np.transpose(X_hat, (1,2,0)), data_range=X.max() - X.min(), multichannel=multichannel)
    test_SSIM = compare_ssim(X, X_hat, data_range=1.0, channel_axis=ch_axis)
    return test_SSIM



def im2patch(im,pch_size,stride=1):
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')
    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')

    C, H, W = im.size()
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = torch.zeros((C, pch_H*pch_W, num_pch)).cuda()
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.view((C, pch_H, pch_W, num_pch))

def chen_estimate(im,pch_size=8):
    """
    Estimated GAT transformed noise to gaussian noise (supposed to be variance 1)
    """
    im=torch.squeeze(im)
    
    #grayscale
    im=im.unsqueeze(0)
    pch=im2patch(im,pch_size,3)
    num_pch=pch.size()[3]
    pch=pch.view((-1,num_pch))
    d=pch.size()[0]
    mu=torch.mean(pch,dim=1,keepdim=True)
    
    X=pch-mu
    sigma_X=torch.matmul(X,torch.t(X))/num_pch
    sig_value,_=torch.symeig(sigma_X,eigenvectors=True)
    sig_value=sig_value.sort().values
    
    
    start=time.time()
    # tensor operation for substituting iterative step.
    # These operation make  parallel computing possiblie which is more efficient

    triangle=torch.ones((d,d))
    triangle= torch.tril(triangle).cuda()
    sig_matrix= torch.matmul( triangle, torch.diag(sig_value)) 
    
    # calculate whole threshold value at a single time
    num_vec= torch.arange(d)+1
    num_vec=num_vec.to(dtype=torch.float32).cuda()
    sum_arr= torch.sum(sig_matrix,dim=1)
    tau_arr=sum_arr/num_vec
    
    tau_mat= torch.matmul(torch.diag(tau_arr),triangle)
    
    # find median value with masking scheme: 
    big_bool= torch.sum(sig_matrix>tau_mat,axis=1)
    small_bool= torch.sum(sig_matrix<tau_mat,axis=1)
    mask=(big_bool==small_bool).to(dtype=torch.float32).cuda()
    tau_chen=torch.max(mask*tau_arr)
      
# Previous implementation       
#    for ii in range(-1, -d-1, -1):
#        tau = torch.mean(sig_value[:ii])
#        if torch.sum(sig_value[:ii]>tau) == torch.sum(sig_value[:ii] < tau):
             #  return torch.sqrt(tau)
#    print('old: ', torch.sqrt(tau))

    return torch.sqrt(tau_chen)

def gat(z,sigma,alpha,g):
    _alpha=torch.ones_like(z)*alpha
    _sigma=torch.ones_like(z)*sigma
    z=z/_alpha
    _sigma=_sigma/_alpha
    f=(2.0)*torch.sqrt(torch.max(z+(3.0/8.0)+_sigma**2,torch.zeros_like(z)))
    return f
def vst(self,transformed,version='MSE'):    
        
        est=chen_estimate(transformed)
        if version=='MSE':
            return ((est-1)**2)
        elif version =='MAE':
            return abs(est-1)
        else :
            raise ValueError("version error in _vst function of train_pge.py")

def inverse_gat(z,sigma1,alpha,g,method='asym'):
   # with torch.no_grad():
    sigma=sigma1/alpha
    if method=='closed_form':
        exact_inverse = ( np.power(z/2.0, 2.0) +
              0.25* np.sqrt(1.5)*np.power(z, -1.0) -
              11.0/8.0 * np.power(z, -2.0) +
              5.0/8.0 * np.sqrt(1.5) * np.power(z, -3.0) -
              1.0/8.0 - sigma**2 )
        exact_inverse=np.maximum(0.0,exact_inverse)
    elif method=='asym':
        exact_inverse=(z/2.0)**2-1.0/8.0-sigma
    else:
        raise NotImplementedError('Only supports the closed-form')
    if alpha !=1:
        exact_inverse*=alpha
    if g!=0:
        exact_inverse+=g
    return exact_inverse

def normalize_after_gat_torch(transformed):
    min_transform=torch.min(transformed)
    max_transform=torch.max(transformed)

    transformed=(transformed-min_transform)/(max_transform-min_transform)
    transformed_sigma= 1/(max_transform-min_transform)
    transformed_sigma=torch.ones_like(transformed)*(transformed_sigma)
    return transformed, transformed_sigma, min_transform, max_transform


def load_img_dict(target_x,target_y,clean_f_num='F64',path_list = ["./dataset_1536x3072_aligned"],
                  test_idx = '_07',debug=True):
    tmp_possible_f_num = ['F01','F02','F04','F08','F16','F32']
    
    x_f_num = [target_x]
    x_f_num.append(clean_f_num)
    print(x_f_num)
    img_dict = {}
    print(os.getcwd())
    if type(path_list) == str:
        path_list = [path_list]
    for data_path in path_list:
        print("=====",data_path, "=====")
            
        f_num_list = sorted(os.listdir(data_path))
        # print(f_num_list)
        for f_num in f_num_list:
            if f_num[0] != 'F':
                continue
            f_path = os.path.join(data_path,f_num)
            # f_number = int(f_num[1:])
            # f_num = f"F{f_number:02d}"
            # if f_num not in x_f_num:
            #     print(f_num , " is not in ", x_f_num)
            #     continue
            img_dict[f_num] = []
            img_list = sorted(os.listdir(f_path))
            img_list = list(filter(lambda x : ".ipynb_checkpoints" not in x,img_list))
            
            for filename in sorted(img_list):
                # if test_idx not in filename:
                #     continue
                img_path = os.path.join(f_path,filename)
                image_idx = int(filename.split("_")[1].split(".")[0])
                real_f_num = filename.split("_")[0]
                filename = f"{real_f_num}_{image_idx:02d}.png"
                img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE) 
                img = np.expand_dims(img,axis=0) / 255.
                img = (img - img.min()) / (img.max() - img.min())
                img_dict[f_num].append(img)
                
                if debug is True:
                    print(filename,f_num,img.shape)
    if debug is True:
        print(img_dict.keys())
        for f_num in img_dict.keys():
            print("   ",len(img_dict[f_num]),img_dict[f_num][0].shape)
    print("====== load img_dict complete ======")
    return img_dict
                        
def dropout_without_energy_perserving(noisy_img,dropout_rate=0.6):
    # mask = np.random.binomial(1, dropout_rate, size=noisy_img.shape).astype(bool)
    # identical code for pytorch
    mask = torch.bernoulli(torch.ones_like(noisy_img)*dropout_rate).to(noisy_img.device)
    return noisy_img * mask


def apply_RPM(noisy_patch,RPM_p=0.6,RPM_type='random',RPM_grid_size=1,RPM_masking_value=0.0):
    
    if RPM_type == 'random':
        mask = np.random.binomial(1, RPM_p, size=noisy_patch.shape).astype(bool)
    elif RPM_type == 'grid':
        mask = np.zeros(noisy_patch.shape).astype(bool)
        num_grid = int(np.prod(noisy_patch.shape) / (RPM_grid_size*RPM_grid_size))
        idx_num_grid = 0
        # mask with box shape
        mask_or_not = np.random.binomial(1, RPM_p, size=num_grid).astype(bool)
        for i in range(0,noisy_patch.shape[0],RPM_grid_size):
            for j in range(0,noisy_patch.shape[1],RPM_grid_size):
                mask[i:i+RPM_grid_size, j:j+RPM_grid_size] = mask_or_not[idx_num_grid]
                idx_num_grid += 1
    elif RPM_type == 'mixed':
        RPM_grid_size = np.random.choice([1,2,4,8,16,32,64])
        if RPM_grid_size == 1:
            mask = np.random.binomial(1, RPM_p, size=noisy_patch.shape).astype(bool)
        else:
            mask = np.zeros(noisy_patch.shape).astype(bool)
            num_grid = int(np.prod(noisy_patch.shape) / (RPM_grid_size*RPM_grid_size))
            idx_num_grid = 0
            # mask with box shape
            mask_or_not = np.random.binomial(1, RPM_p, size=num_grid).astype(bool)
            for i in range(0,noisy_patch.shape[0],RPM_grid_size):
                for j in range(0,noisy_patch.shape[1],RPM_grid_size):
                    mask[i:i+RPM_grid_size, j:j+RPM_grid_size] = mask_or_not[idx_num_grid]
                    idx_num_grid += 1

    else:
        raise NotImplementedError(f"RPM_type has to be random or grid, not {RPM_type}")
    if type(noisy_patch) == torch.Tensor:
        mask = torch.tensor(mask)
    noisy_patch[mask] = RPM_masking_value
    return noisy_patch
import PIL
def save_png(img,path):
    if type(img) == torch.Tensor:
        img = img.cpu().numpy()
    # img = (img-img.min())/(img.max()-img.min())
    if img.max() > 1:
        img = img/255
    img = PIL.Image.fromarray((img*255).astype('uint8'))
    
    img.save(path, format='png')
def save_eps(img,path):
    if type(img) == torch.Tensor:
        img = img.cpu().numpy()
    # img = (img-img.min())/(img.max()-img.min())
    if img.max() > 1:
        img = img/255
    img = PIL.Image.fromarray((img*255).astype('uint8'))
    img.save(path,dpi=(300,300),mode='EPS')