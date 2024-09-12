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

# load whole image[size : (2048, 3072)] in "whole_images" 
with open(f"./info_img_path.txt",'r') as f:
    data_path = f.read().split('\n')[0]
print(data_path)
crop_size = 256

search_range = 40
shift_info = {}
show_log = True
print("search_range : ",search_range)
print("==== make align information =====")
## make align information

# noisy_f_num = ['F01','F02','F04','F08','F16','F32']
f_num_list = sorted(os.listdir(data_path)) # ['F01_1', 'F08_2', 'F64_2']
noisy_f_num = sorted(list(map(lambda x : x.split('_')[0],os.listdir(data_path))))
# clean_f_num = 'F64'
clean_f_num = noisy_f_num[-1] # ex) 'F64'
noisy_f_num = noisy_f_num[:-1] # ex) ['F01','F08']
for i,f_num in enumerate(f_num_list): 
    original_f_num = f_num.split('_')[0]
    if original_f_num == clean_f_num:
        continue
    # for f_num in [f"{original_f_num}_1",f"{original_f_num}_2"]:
    f_idx = f_num.split('_')[1]
    for image_idx, image_path in tqdm(enumerate(sorted(glob.glob(f"{data_path}/{f_num}/*.png")))):
        #print(image_path)
        file_name = image_path.split('/')[-1]
        img_info = file_name.split('.')[0].split('_')
        f_num, image_idx = img_info
        clean_path = os.path.join(data_path,f"{clean_f_num}_2/{clean_f_num}_{image_idx}.png")
        if show_log is True : 
            print(image_path, clean_path)
        
        im = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        target_im = cv2.imread(clean_path,cv2.IMREAD_GRAYSCALE)
        print(im.shape, target_im.shape)
        v_shift, h_shift = get_shift_info(im, target_im,v_width=search_range,h_width=search_range)
        shift_info[image_path] = {'v_shift' : v_shift, 'h_shift' : h_shift}
        if show_log is True:
            print( v_shift,h_shift)
shift_info['search_range'] = search_range
save_file_name = "./shift_info.txt"
with open(save_file_name, 'w') as f:
    f.write(json.dumps(shift_info,indent="\t"))
print("===========================")
print(f"write to {save_file_name} complete")
