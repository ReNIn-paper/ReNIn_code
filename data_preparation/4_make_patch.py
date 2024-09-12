#!/usr/bin/env python
# coding: utf-8

from ast import parse
import torch
import os,glob,sys
import h5py
import argparse
#from multiprocessing import Process, Lock
import torch.multiprocessing as mp
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
from core.utils import seed_everything
from core.patch_generate_6_8 import make_dataset_iterative_6_8

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--crop-size', type=int, default=256)
parser.add_argument('--clean-f-num', type=str, default='F64')
parser.add_argument('--pad', type=int, default=128)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--len-training-patch', type=int, default=21600)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= "2" 
seed = args.seed
seed_everything(seed) # Seed 고정

with open(f"./info_img_path.txt",'r') as f:
   data_path = f.read().split('\n')[0]
   img_size = data_path.split('_')[-1]
   img_size = tuple(map(int,img_size.split('x')))
data_path += "_aligned"
print(data_path,img_size)

len_training_patch = args.len_training_patch#21600

# read image list & check image length
crop_size = args.crop_size
image_list = []
image_len = None
for f_num in os.listdir(data_path):
   if f_num.startswith('F'):
      print(f_num)
      image_list.append(sorted(glob.glob(f"{data_path}/{f_num}/*.png")))
for idx in range(len(image_list)-1):
   f_image_list = image_list[idx]
   image_len = len(f_image_list)
   f_image_list_next = image_list[idx+1]
   assert len(f_image_list) == len(f_image_list_next), f"len(f_image_list) != len(f_image_list_next) :\
      {len(f_image_list)} != {len(f_image_list_next)}"
# calculated num_crop 
print(f"image_len : {image_len}")
num_crop = int(len_training_patch/(image_len-1))
if len_training_patch % (image_len-1) != 0:
   num_crop += 1
print(f"num_crop : {num_crop}")
print(f"expected training len : {num_crop * (image_len-1)}")
expected_length_training_patch = num_crop * (image_len-1)
    

m = mp.Manager()
write_lock = m.Lock()

f_num_list = sorted(os.listdir(data_path)) # ['F01_1', 'F08_2', 'F64_2']

make_dataset_iterative_6_8(data_path,img_size, write_lock,image_len = image_len,num_crop=num_crop,crop_size=args.crop_size, 
                       f_num_list=f_num_list, pad = args.pad, is_test=args.test)
   
print("All Complete")
#print(f"shift info : {len(shift_info['SET1'].keys())}")





