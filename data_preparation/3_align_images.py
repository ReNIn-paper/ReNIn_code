import torch
import torch.backends.cudnn as cudnn
import torchvision
import cv2 
from matplotlib import pyplot as plt
import os,glob,sys
from tqdm import tqdm
import numpy as np
import PIL
import random
from PIL import Image
import h5py
from typing import Generator
import gc,json 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 1 to use
from core.utils import seed_everything
from core.patch_generate import *

seed_everything(0) # Seed 고정

with open(f"./info_img_path.txt",'r') as f:
    data_path = f.read().split('\n')[0]
    img_size = data_path.split('_')[-1]
    img_size = tuple(map(int,img_size.split('x')))
print(data_path,img_size)

crop_size = 256

show_log = True
f_num_list = sorted(os.listdir(data_path)) # ['F01_1', 'F08_2', 'F64_2']
noisy_f_num = sorted(list(map(lambda x : x.split('_')[0],os.listdir(data_path))))
clean_f_num = noisy_f_num[-1] # ex) 'F64'
noisy_f_num = noisy_f_num[:-1] # ex) ['F01','F08']

## alignment correction

aligned_data_path=f"{data_path}_aligned"
os.makedirs(aligned_data_path,exist_ok=True)
for f_num in f_num_list:
    os.makedirs(os.path.join(aligned_data_path,f_num),exist_ok=True)

import json

shift_info = {}
save_file_name = "./shift_info.txt"
with open(save_file_name,'r') as f:
    shift_info = json.load(f)
print(shift_info.keys())
print("==== align image =====")

try:
    search_range = shift_info['search_range']
    del shift_info['search_range']
except :
    print("search_range is not exist")
    search_range = 40
pad = search_range+1 # if search_range is 40, pad is 41 


for img_path, value in shift_info.items():
    if "checkpoints" in img_path :
        continue
    file_name = img_path.split('/')[-1]
    f_num = img_path.split('/')[-2]
    f_path = os.path.join(aligned_data_path,f_num)
    os.makedirs(f_path,exist_ok=True)
    save_path = os.path.join(f_path,file_name)
    print(img_path)
    if os.path.exists(img_path) is False:
        raise ValueError(f"{img_path} is not exist")
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    
    print(f"{file_name} save path : {save_path}")

    v_shift, h_shift = shift_info[img_path].values()
    padded_img = img[pad+v_shift:-pad+v_shift:,pad+h_shift:-pad+h_shift]
    
    # assert padded_img.shape == (1454,2990), f"{padded_img.shape}"
    print(padded_img.shape)
    assert padded_img.shape[0] != 0 and padded_img.shape[1] != 0, f"image size is zero,{padded_img.shape}"
    print(file_name,v_shift, h_shift,padded_img.shape)
    
    cv2.imwrite(save_path,padded_img)
print("=== start clean image ===")
# done with clean path
for f_idx in [1,2]:
    for img_path in glob.glob(f"{data_path}/F64_{f_idx}/*"):
        print(img_path)
        file_name = img_path.split('/')[-1]
        f_num = img_path.split('/')[-2]
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        padded_img = img[pad:-pad,pad:-pad]
        # assert padded_img.shape == (1454,2990), f"{padded_img.shape}"
        # check whether one of image shape is zero
        assert padded_img.shape[0] != 0 and padded_img.shape[1] != 0, f"image size is zero,{padded_img.shape}"
            
        f_path = os.path.join(aligned_data_path,f_num)
        os.makedirs(f_path,exist_ok=True)
        save_path = os.path.join(f_path,file_name)
        cv2.imwrite(save_path,padded_img)
        print(file_name, padded_img.shape)

print("===========================")
print("complete align")