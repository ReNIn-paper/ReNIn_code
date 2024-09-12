import torch
from torch.utils.data import DataLoader

import numpy as np
import os
import sys
sys.path.append(os.path.abspath("../../core"))
from datetime import date
from .utils import get_PSNR, get_SSIM, inverse_gat, gat, normalize_after_gat_torch
from .loss_functions import mse_bias, mse_affine, emse_affine
from .models import New_model
from .fcaide import FC_AIDE
from .dbsn import DBSN_Model
from .unet import est_UNet
from .model_NAFNet import NAFNet
import h5py
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark=True

class produce_denoised_img_no_crop(object):
    """
    produce denoised image with pretrained weights
    """
    def __init__(self,_pge_weight_dir=None,_fbi_weight_dir=None, _save_file_name = None, _args = None):
        
        self.args = _args
        if self.args.loss_function == 'EMSE_Affine':
            assert self.args.pge_weight_dir != None, "Need pretrained PGE weight directory information"
        
            # self.pge_weight_dir = '../weights/' + self.args.pge_weight_dir
            self.pge_weight_dir =  self.args.pge_weight_dir
        

        self.result_psnr_arr = []
        self.result_ssim_arr = []
        self.result_time_arr = []
        self.result_denoised_img_arr = []
        
        self.save_file_name = _save_file_name
        self.date = date.today().isoformat()
        num_output_channel = 1
        if self.args.loss_function[:10] == 'MSE_Affine': # MSE_Affine
            self.loss = mse_affine
            num_output_channel = 2
            self.args.output_type = "linear"
        elif self.args.loss_function == 'N2V': #lower bound
            self.loss = mse_bias
            num_output_channel = 1
            self.args.output_type = "linear"
        elif self.args.loss_function[:3] == 'MSE': # MSE
            self.loss = torch.nn.MSELoss()
            num_output_channel = 1
            self.args.output_type = "linear"
        elif self.args.loss_function == 'EMSE_Affine':
            
            self.loss = emse_affine
            num_output_channel = 2
            
            ## load PGE model
            self.pge_model=est_UNet(num_output_channel,depth=3)
            self.pge_model.load_state_dict(torch.load(self.pge_weight_dir))
            self.pge_model=self.pge_model.cuda()
            
            for param in self.pge_model.parameters():
                param.requires_grad = False
            
            self.args.output_type = "sigmoid"
        
        if self.args.model_type == 'FC-AIDE':
            self.model = FC_AIDE(channel = 1, output_channel = num_output_channel, filters = 64, num_of_layers=10, output_type = self.args.output_type, sigmoid_value = self.args.sigmoid_value)
        elif self.args.model_type == 'DBSN':
            self.model = DBSN_Model(in_ch = 1,
                            out_ch = num_output_channel,
                            mid_ch = 96,
                            blindspot_conv_type = 'Mask',
                            blindspot_conv_bias = True,
                            br1_block_num = 8,
                            br1_blindspot_conv_ks =3,
                            br2_block_num = 8,
                            br2_blindspot_conv_ks = 5,
                            activate_fun = 'Relu')
        elif self.args.model_type == 'NAFNet_light':
            self.model =  NAFNet(img_channel=1,output_channel= num_output_channel,width=8, middle_blk_num=4,
                      enc_blk_nums=[2,2,4,4], dec_blk_nums=[2,2,2,2]) # 809257
        else:
            self.model = New_model(channel = 1, output_channel =  num_output_channel, filters = self.args.num_filters, num_of_layers=self.args.num_layers, case = self.args.model_type, output_type = self.args.output_type, sigmoid_value = self.args.sigmoid_value)
            
        self.model.load_state_dict(torch.load(_fbi_weight_dir))
        self.model = self.model.cuda()
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2, 3])
        
    def get_X_hat(self, Z, output):

        X_hat = output[:,:1] * Z + output[:,1:]
            
        return X_hat
        
    def eval(self,image,keyword=None,ablated_index=None):
        """
        Evaluates denoiser on validation set.
        image : (1474, 3010) SEM_semi image        
        """

        psnr_arr = []
        ssim_arr = []
        # image = torch.Tensor(image/255.).cuda() # flaot 32
        image = torch.Tensor(image).cuda() # float 32
        # print(image.dtype)
        
        with torch.no_grad():
            # print(imgs.shape[0]*256**2,torch.count_nonzero(imgs))
            image = image.cuda()
            
            if self.args.loss_function =='EMSE_Affine':
                est_param=self.pge_model(image)
                original_alpha=torch.mean(est_param[:,0])
                original_sigma=torch.mean(est_param[:,1])
                    
                transformed=gat(image,original_sigma,original_alpha,0)
                transformed, transformed_sigma, min_t, max_t= normalize_after_gat_torch(transformed)
                
                transformed_target = torch.cat([transformed, transformed_sigma], dim = 1)
#                     target = torch.cat([target,transformed_sigma], dim = 1)

                output = self.model(transformed)
            else :
                output = self.model(image)
            if self.args.debug is True:
                print("output shape ",output.shape)
            
            ## affine mapping에 맞게 이미지를 바꾸어 준다.
            if self.args.loss_function[:10] == 'MSE_Affine':
                X_hat = self.get_X_hat(image,output).cpu().numpy()
                
                
            elif self.args.loss_function[:3] == 'MSE':
                output = output.cpu().numpy()
                X_hat = np.clip(output, 0, 1)
            
            elif self.args.loss_function == 'N2V':
                X_hat = np.clip(output.cpu(), 0, 1)
            # X_hat = np.clip(self.get_X_hat(image,output).cpu(), 0, 1)

            elif self.args.loss_function == 'EMSE_Affine':
                
                transformed_Z = transformed_target[:,:1]
                X_hat = self.get_X_hat(transformed_Z,output).cpu().numpy()
                
                transformed=transformed.cpu().numpy()
                original_sigma=original_sigma.cpu().numpy()
                original_alpha=original_alpha.cpu().numpy()
                min_t=min_t.cpu().numpy()
                max_t=max_t.cpu().numpy()
                X_hat =X_hat*(max_t-min_t)+min_t
                X_hat=np.clip(inverse_gat(X_hat,original_sigma,original_alpha,0,method='closed_form'), 0, 1)
 

            # if self.args.debug is True:
            #     print(output.shape)
            #     if 'Affine' in self.args.loss_function:
            #         if type(output) == torch.Tensor:
            #             output = output.cpu().detach().numpy()
            #         plt.subplot(121)
            #         plt.imshow(output[0][0])
            #         plt.subplot(122)
            #         plt.imshow(output[0][1])
            #     else :
            #         pass
            #     plt.pause(0.1)
            #     print("X_hat shape ",X_hat.shape)

            # # denoising is completed
            # if self.args.debug is True:
            #     print(X_hat.shape)
            #     if type(X_hat) == torch.Tensor:
            #         X_hat = X_hat.cpu().detach().numpy()
            #     plt.imshow(X_hat[0][0])
            #     plt.pause(0.1)
            return X_hat
            