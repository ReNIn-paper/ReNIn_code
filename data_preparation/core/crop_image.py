import numpy as np
import os 
import matplotlib.pyplot as plt

def scale_f_num_0to3(f_num : str):
    f_index = int(np.log2(int(f_num.split("F")[-1]))-3)
    return f_index
def crop_image(image : np.array ,crop_size : int = 256, 
               x_start_index : int=0, y_start_index : int=0, 
               img_write : bool = False):
    image_size = image.shape
    # print(image_size)
    num_x_axis = (image_size[0]-x_start_index)//crop_size
    num_y_axis = (image_size[1]-y_start_index)//crop_size
    x_axis = [i for i in range(x_start_index,image_size[0]-x_start_index,crop_size)]
    y_axis = [i for i in range(y_start_index,image_size[1]-y_start_index,crop_size)]
    cropped_images = np.zeros([num_x_axis*num_y_axis,crop_size,crop_size])
    index = 0
    # print(f"make {len(x_axis)*len(y_axis)} images ({len(x_axis)} x {len(y_axis)})")
    # make crop image per index
    for x in x_axis:
        for y in y_axis:
            cropped_images[index] = image[x:x+crop_size,y:y+crop_size]
            index +=1
            if img_write is True:
                f_name : str = f'{set_folder}/{filename.split(".")[0]}_{i:03d}.png'
                plt.imsave(f_name,cropped_images[i])   
    return cropped_images
def make_image_crop(images : dict, image_size : list, crop_size : int = 256, img_write : bool = False):
    # images dict structure
    # Set number -> F number 
    total_cropped_image = dict()
    num_x_axis = image_size[0]//crop_size
    num_y_axis = image_size[1]//crop_size
    for set_num in images.keys():
        # print(set_num)
        trim_set_num = set_num.replace(" ", "")[1:-1] # [SET 1] -> SET1
        total_cropped_image[trim_set_num] = np.zeros([4,173*16,crop_size,crop_size])
        #print(total_cropped_image[set_num][0].shape)
        if img_write is True :
            set_folder = os.path.join(main_folder,set_num)
            os.makedirs(set_folder,exist_ok=True)
        for f_num in images[set_num].keys():
            print(set_num + "->" + f_num)
            f_index = scale_f_num_0to3(f_num)
            for filename in whole_images[set_num][f_num]:
                image_index = int(filename.split("_")[0])
                image = whole_images[set_num][f_num][filename]
                
                cropped_images_1 = crop_image(image, crop_size, 0,0,img_write=False) # [F#,img# * 16 + 0~95, img]
                cropped_images_2 = crop_image(image, crop_size, 128,128,img_write=False) # [F#,img# * 16 + 96~172, img]
                cropped_images = np.vstack((cropped_images_1,cropped_images_2))
                
                i_from, i_to = cropped_images.shape[0] * (image_index-1), cropped_images.shape[0] * (image_index)
                #print(f"total_cropped_image[{i_from} ~ {i_to}] save image, shape : {cropped_images.shape} vs {total_cropped_image[trim_set_num][f_index].shape}")
                total_cropped_image[trim_set_num][f_index][i_from:i_to] = cropped_images
    return total_cropped_image

