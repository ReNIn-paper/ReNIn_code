import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import scipy.io as sio
import os 

from .utils import TedataLoader, chen_estimate, get_PSNR, get_SSIM, inverse_gat, gat, normalize_after_gat_torch
from .unet import est_UNet

import time

torch.backends.cudnn.benchmark=True

class Test_PGE(object):
    def __init__(self,_te_data_dir=None,_pge_weight_dir=None,_save_file_name = None, _args = None):
        
        self.args = _args
        self.te_data_loader = TedataLoader(_te_data_dir, self.args)
        self.te_data_loader = DataLoader(self.te_data_loader, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        self.result_alpha_arr = []
        self.result_beta_arr = []
        self.result_time_arr = []
        self.save_file_name = _save_file_name

        ## load PGE model
        
        num_output_channel = 2
        self.pge_model=est_UNet(num_output_channel,depth=3)
        self.pge_model.load_state_dict(torch.load(_pge_weight_dir))
        self.pge_model.cuda()
        
    def eval(self):
        """Evaluates denoiser on validation set."""
        print(f"data name : {self.args.data_name}")
        alpha_arr = []
        beta_arr = []
        time_arr = []
        estimated_gaussian_noise_level_arr = []
        with torch.no_grad():

            for batch_idx, (source, target) in enumerate(self.te_data_loader):

                start = time.time()

                source = source.cuda()
                target = target.cuda()
                
                # Denoise
                est_param=self.pge_model(source)
                
                original_alpha=torch.mean(est_param[:,0])
                original_beta=torch.mean(est_param[:,1])
                
                inference_time = time.time()-start
                
                print(f"image : {batch_idx:02d} ->\t alpha : {round(float(original_alpha),10)}, sigma : {round(float(original_beta),10)} ")
                transformed_with_pge = gat(source,original_beta,original_alpha,0)
                estimated_gaussian_noise_level = chen_estimate(transformed_with_pge)
                print(f"estimated gaussian_noise  : {estimated_gaussian_noise_level}")
                transformed_with_pge, transformed_sigma, min_t, max_t= normalize_after_gat_torch(transformed_with_pge)
                
                
                original_beta=original_beta.cpu().numpy()
                original_alpha=original_alpha.cpu().numpy()

                estimated_gaussian_noise_level_arr.append(estimated_gaussian_noise_level.cpu().numpy())
                alpha_arr.append(original_alpha)
                beta_arr.append(original_beta)
                time_arr.append(inference_time)
        mean_alpha = np.mean(alpha_arr)
        mean_beta = np.mean(beta_arr)
        mean_time = np.mean(time_arr)
        alpha_sigma = np.stack((alpha_arr,beta_arr,estimated_gaussian_noise_level_arr),axis=1)
        os.makedirs('./output_parameter',exist_ok=True)
        np.savetxt(f'./output_parameter/{self.save_file_name}_alpha_sigma_estimated_gaussian.txt',alpha_sigma,delimiter='\t')
        print ('Mean(alpha) : ', round(mean_alpha,10), 'Mean(beta) : ', round(mean_beta,10))
        f = open("./result.txt",'a')
        print(f"{self.args.data_name} {self.args.dataset_type} alpha {self.args.alpha} beta  {self.args.beta}",file=f)
        if self.args.test_alpha != 0:
            print(f"test alpha {self.args.test_alpha} test beta  {self.args.test_beta}",file=f)
        print(f"Mean estimated_gaussian_noise_level : {np.mean(estimated_gaussian_noise_level_arr)}",file=f)
        f.close()

        # metric = pd.DataFrame(index = len(alpha_arr), columns = ['alpha','sigma','estimated_gaussian'])
        # metric['alpha'] = alpha_arr
        # metric['beta'] = beta_arr
        # metric['estimated_gaussian'] = estimated_gaussian_noise_level_arr
        # save_metric_name = f'./result_data/{self.save_file_name}_alpha_sigma_estimated_gaussian.csv'
        # metric.to_csv(save_metric_name)     
        return 
  


