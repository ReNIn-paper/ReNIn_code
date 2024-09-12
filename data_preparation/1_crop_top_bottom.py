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
data_path = "../dataset"

def sem_generator(data_path : str):
    for f_num in sorted(os.listdir(data_path)):
        f_path = os.path.join(data_path,f_num)
        f_info = f_num.split("_")
        f_number = int(f_info[0][1:])
        try:
            f_idx = int(f_info[1])
        except:
            f_idx = 1
        for file_name in sorted(os.listdir(f_path)):
            print(file_name)
            file_path = os.path.join(f_path,file_name)
            # print(f_num)
            img_info = file_name.split('.')[0].split('_')
            f_num, image_idx = img_info
            
            
            img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
            
            yield [f"F{f_number:02d}",f_idx, int(image_idx),img]

def generate_top_bottom_crop_image(data_path : str,generator : Generator[str,int,np.array],pad=256,save=True):
    # save_dir = data_path+"_1536x3072"
    save_dir = None
    for img in generator:
        # print(img[:3])
        f_num,f_idx, image_idx = img[:3]
    
        img = img[-1][pad:-pad,:]
        if save_dir is None:
            img_shape = f"{img.shape[0]}x{img.shape[1]}"
            save_dir = data_path+f"_{img_shape}"
            os.makedirs(save_dir,exist_ok=True)
            with open(f"./info_img_path.txt",'w') as f:
                f.write(f"{save_dir}\n")
        #print(img.shape) # (1792, 3072)
        #plt.imshow(img,cmap='gray')
        file_name = f"{f_num}_{image_idx:02d}.png"
        f_path = os.path.join(save_dir,f"{f_num}_{f_idx}")
        os.makedirs(f_path,exist_ok=True)
        file_path = os.path.join(f_path,file_name)
        # print(file_name)
        if save is True:
            print(f"{file_path} is saved")
            cv2.imwrite(file_path,img)
        
    return save_dir

# crop top&bottom of original image
print("==== load original image =====")
gen = sem_generator(data_path)
print("==== crop top&bottom of original image =====")
data_path = generate_top_bottom_crop_image('../dataset',gen,pad=256,save=True)
