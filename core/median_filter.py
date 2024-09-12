import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
import time
from scipy.ndimage import median_filter
import numpy as np
from copy import deepcopy

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            # print(x.size())
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        # print(type(x),x.dtype,x.shape)
        # only works for flaot, not uint8
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x
class median_filter_architecture(nn.Module):
    def __init__(self, MedianPool2d_layer, repeat=3):
        super(median_filter_architecture, self).__init__()
        self.layer = MedianPool2d_layer
        self.repeat = repeat
    def forward(self,x):
        for i in range(self.repeat):
            x = self.layer(x)
        return x
def init_median_filter_model(kernel_size =11, repeat : int = 3):
    layer = MedianPool2d(kernel_size=kernel_size,same=True)
    model = median_filter_architecture(layer, repeat=repeat)
    return model

def apply_median_filter_gpu_simple_keep_gpu(img,kernel_size=11, repeat : int = 3,time_log=False):
    img = torch.Tensor(img).cuda()
    model = init_median_filter_model(kernel_size=kernel_size, repeat=repeat).cuda()
    while len(img.size()) < 4 :
        img = torch.unsqueeze(img,dim=0) # expand_dims(img,axis=0)
    while len(img.size()) > 4 :
        img = torch.squeeze(img,dim=0) # expand_dims(img,axis=0)
    img = model(img)
    return img[0]
def apply_median_filter_gpu_simple(img,kernel_size=11, repeat : int = 3,time_log=False):
    model = init_median_filter_model(kernel_size=kernel_size, repeat=repeat).cuda()
    while len(img.shape) <= 3 :
        img = np.expand_dims(img,axis=0)
    img = torch.Tensor(img).cuda()
    # print(img.shape)
    if time_log is True:
        present_time = time.time()
    img = model(img)
    if time_log is True:
        last_time = time.time()
        print(f"=== Applying median filter : {last_time - present_time} s===")
    return img.cpu().numpy()
def apply_median_filter_gpu(model,img,time_log=False):
    img = torch.Tensor(img).cuda()
    if time_log is True:
        present_time = time.time()
    img = model(img)
    if time_log is True:
        last_time = time.time()
        print(f"=== Applying median filter : {last_time - present_time} s===")
    return img
def apply_median_filter_cpu(img : np.array ,kernel_size  : tuple = (11,11), repeat: int =3 ,plot : bool =False):
    out = deepcopy(img)
    if plot is True:
        plt.title(f'before iter')
        plt.imshow(out[:200,:200])
        plt.pause(0.01)
        plt.figure(figsize=(20,4))
        plt.plot(out[50][:200])
        plt.pause(0.01)
    for i in range(repeat):
        out = median_filter(out,kernel_size)
        if plot is True:
            plt.title(f'{i+1} iter')
            plt.imshow(out[:200,:200])
            plt.pause(0.01)
            plt.figure(figsize=(20,4))
            plt.plot(out[50][:200])
            plt.pause(0.01)
    return out