import os
import torch
import random
import numpy as np
from contextlib import contextmanager
from torch.multiprocessing import Pool
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

@contextmanager
def poolcontext(*args, **kwargs):
    pool = Pool(*args, **kwargs)
    yield pool
    pool.terminate()
def get_metric(ref_img,img):
    psnr = peak_signal_noise_ratio(ref_img,img,data_range=1.0)
    ssim = structural_similarity(ref_img,img,data_range=1.0)
    return {'PSNR' : psnr, 'SSIM': ssim}

def convert_size(size_bytes):
    import math
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%.2f %s" % (s, size_name[i])
def get_number_from_MB(converted_size : str)-> float:
    return float(converted_size.split(" ")[0])
def show_file_info():
    data_path = "../dataset"
    for set_num in sorted(os.listdir(data_path)):
        set_path = os.path.join(data_path,set_num)
        cnt = 0

        for f_num in sorted(os.listdir(set_path)):
            f_path = os.path.join(set_path,f_num)
            avg_info = {"lowest_size" : -1, "mean_size" : 0, "cnt" : 0, "highest_size" : -1}
            for file_name in os.listdir(f_path):
                file_path = os.path.join(f_path,file_name)
                file_size = os.path.getsize(file_path)
                #print(file_name,file_size)

                if file_size == 0:
                    print(f"{file_name} has 0 bytes")
                else :
                    avg_info["cnt"] +=1
                    formatted_size = convert_size(file_size)
                    avg_info["mean_size"] += get_number_from_MB(formatted_size)
                    if avg_info["lowest_size"] == -1 or get_number_from_MB(formatted_size) < get_number_from_MB(avg_info["lowest_size"]):
                        avg_info["lowest_size"] = formatted_size
                    elif avg_info["highest_size"] == -1 or get_number_from_MB(formatted_size) > get_number_from_MB(avg_info["highest_size"]):
                        avg_info["highest_size"] = formatted_size
            avg_info["mean_size"] = str(round(avg_info["mean_size"] / avg_info["cnt"],2)) + " MB"
            print(set_num,f_num,"\t",avg_info)
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def get_dict_structure(target : dict):
    print("=== dict structure ===")
    sub_dict = target
    while(True):
        try :
            for keys in sub_dict.keys():
                print(keys, end=", ")
            sub_dict = sub_dict[keys]
            print("")
        except : 
            break
def show_two_images(img1,img2):
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.title(f"img1")
    plt.imshow( img1,cmap='gray')

    plt.subplot(1,2,2)
    plt.axis('off')
    plt.title(f"img2 ")
    plt.imshow( img2,cmap='gray')
def visualize_imgs(imgs : list, titles : dict):
    n = len(imgs)
    for i in range(n):
        index = i+1
        plt.subplot(1,n,index)
        plt.axis('off')
        plt.title(titles[index])
        plt.imshow(imgs[i],cmap='Spectral')