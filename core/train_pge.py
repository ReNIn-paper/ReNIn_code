import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torchvision.transforms.functional as tvF
import scipy.interpolate as sip
import wandb
from .utils import TrdataLoader, TedataLoader, get_PSNR, get_SSIM, chen_estimate, gat,normalize_after_gat_torch,TedataLoader_FullImage
import torchvision as vision
import sys
from .unet import est_UNet  
from .logger import Logger
import time
import sys, os
from tqdm import tqdm
from torchvision import transforms

def select_data_type(_tr_data_dir, _te_data_dir, x_avg_num, y_avg_num,args):
    if 'SEM_semi' in args.data_name:
        print(_tr_data_dir)
        print(_te_data_dir)
        tr_data_loader = TrdataLoader(_tr_data_dir, args)
        tr_data_loader = DataLoader(tr_data_loader, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        te_data_loader = TedataLoader(_te_data_dir, args)
        te_data_loader = DataLoader(te_data_loader, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    else :
        raise NotImplementedError("Not implemented data type", args.data_type, args.data_name)
    return tr_data_loader, te_data_loader
class Train_PGE(object):
    def __init__(self,_tr_data_dir=None, _te_data_dir=None, _save_file_name = None, _args = None):
        
        self.tr_data_dir = _tr_data_dir
        self.te_data_dir = _te_data_dir
        self.args = _args
        self.save_file_name = _save_file_name
        
        self.tr_data_loader, self.te_data_loader = select_data_type(_tr_data_dir, _te_data_dir, \
                                                                    self.args.x_f_num, self.args.y_f_num,self.args)

        self.result_tr_loss_arr = []

        self.logger = Logger(self.args.nepochs, len(self.tr_data_loader))
        self._compile()

    def _compile(self):
        
        self.model=est_UNet(2,depth=self.args.unet_layer)
        
        pytorch_total_params = sum([p.numel() for p in self.model.parameters()])
        print ('num of parameters : ', pytorch_total_params)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, self.args.drop_epoch, gamma=self.args.drop_rate)
        
        self.model = self.model.cuda()

       
    def save_model(self, epoch):
        torch.save(self.model.state_dict(), './weights/'+self.save_file_name  + '.w')
        return
    
        
    def eval(self):
        """Evaluates denoiser on validation set."""        
        
        name_sequence = []
        if self.args.data_name == "SEM_semi":
            for i in range(1,11):
                set_num = f"SET{i:02d}_"
                if i < 5:
                    f_num_list = map(lambda x : f"{set_num}{x}",['F08', 'F16', 'F32'])
                else :
                    f_num_list = map(lambda x : f"{set_num}{x}",['F01', 'F02', 'F04', 'F08', 'F16', 'F32'])
                name_sequence += f_num_list
        loss_arr = []
        a_arr=[]
        b_arr=[]
        estimated_gaussian_noise_level_arr = []
        with torch.no_grad():
            for batch_idx, (source, target) in enumerate(self.te_data_loader):
                source = source.cuda()
                target = target.cuda()
                # Denoise
                
                noise_hat = self.model(source)
  
                # target = target.cpu().numpy()
                # noise_hat = noise_hat.cpu().numpy()

                target = target.cpu().numpy()
                noise_hat = noise_hat.cpu()

                predict_alpha=torch.mean(noise_hat[:,0])
                predict_sigma=torch.mean(noise_hat[:,1])
                transformed_with_pge = gat(source,predict_sigma,predict_alpha,0)
                loss=self._vst(transformed_with_pge,self.args.vst_version)
                estimated_gaussian_noise_level = chen_estimate(transformed_with_pge)
                
                loss_arr.append(loss.item())
                a_arr.append(predict_alpha.item())
                b_arr.append(predict_sigma.item())
                estimated_gaussian_noise_level_arr.append(estimated_gaussian_noise_level.item())

                if batch_idx % 50 == 0 and self.args.log_off is False:
                    name_str = f""
                    if self.args.data_name == "SEM_semi":
                        name_str += f"{name_sequence[batch_idx // 50]}"

                if batch_idx % 50 == 49 and self.args.log_off is False:
                    self.args.logger.log({f"eval_{name_str}/loss" : np.mean(loss_arr[-50:])})
                    self.args.logger.log({f"eval_{name_str}/alpha" : np.mean(a_arr[-50:])})
                    self.args.logger.log({f"eval_{name_str}/sigma" : np.mean(b_arr[-50:])}) 
                    self.args.logger.log({f"eval_{name_str}/estimated_gaussian_noise_level" : np.mean(estimated_gaussian_noise_level_arr[-50:])})
        mean_te_loss = np.mean(loss_arr)
        return mean_te_loss, a_arr,b_arr ,estimated_gaussian_noise_level_arr
    
    def _on_epoch_end(self, epoch, mean_tr_loss):
        """Tracks and saves starts after each epoch."""
        self.save_model(epoch)
        mean_te_loss, a_arr, b_arr,estimated_gaussian_noise_level_arr = self.eval()
        self.result_tr_loss_arr.append(mean_tr_loss)
        self.args.logger.log({
                'train/mean_loss'   : mean_tr_loss,
                'test/mean_loss'    : mean_te_loss,
                'alpha'         : a_arr,
                'sigma'         : b_arr,
                'estimated_gaussian_noise_level' : estimated_gaussian_noise_level_arr
        })
    def _vst(self,transformed,version='MSE'):    
        
        est=chen_estimate(transformed)
        if version=='MSE':
            return ((est-1)**2)
        elif version =='MAE':
            return abs(est-1)
        else :
            raise ValueError("version error in _vst function of train_pge.py")
     
    def train(self):
        """Trains denoiser on training set."""
        num_batches = len(self.tr_data_loader)
        for epoch in range(self.args.nepochs):
            self.scheduler.step()
            tr_loss = []

            for batch_idx, (source, target) in enumerate(self.tr_data_loader):
                self.optim.zero_grad()
                source = source.cuda()
                target = target.cuda()
                
                noise_hat=self.model(source)
                
                predict_alpha=torch.mean(noise_hat[:,0])
                predict_sigma=torch.mean(noise_hat[:,1])

                predict_gat=gat(source,predict_sigma,predict_alpha,0)     

                
                loss=self._vst(predict_gat,self.args.vst_version)
                loss.backward()
                self.optim.step()  

                self.logger.log(losses = {'loss': loss, 'pred_alpha': predict_alpha, 'pred_sigma': predict_sigma}, lr = self.optim.param_groups[0]['lr'])
                tr_loss.append(loss.detach().cpu().numpy())
                if self.args.batch_test is True and batch_idx > 10:
                    break
                
            mean_tr_loss = np.mean(tr_loss)
            # print("on epoch end")
            self._on_epoch_end(epoch+1, mean_tr_loss)    
            if self.args.batch_test is True:
                
                sys.exit(0)

            if self.args.test is True and epoch > 2:
                print("==== 3 epoch test end ====")
                sys.exit(0)

