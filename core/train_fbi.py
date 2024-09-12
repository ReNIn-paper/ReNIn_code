import os
from statistics import median
from matplotlib.pyplot import step
from sklearn import manifold
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import wandb
import numpy as np
import scipy.io as sio
from datetime import date
from .utils import TedataLoader, TrdataLoader, apply_RPM, get_PSNR, get_SSIM, inverse_gat, gat, normalize_after_gat_torch,TedataLoader_FullImage
from .loss_functions import mse_bias, mse_affine, emse_affine
from .models import New_model, New_model_tiny
from .fcaide import FC_AIDE
from .dbsn import DBSN_Model
from .unet import est_UNet
from .model_UNet import UNet
from .model_NAFNet import NAFNet
from .model_RED import REDNet30
from .model_UNet_n2n import N2N_UNet
from .models_dropout import *
from .model_UNet_n2n_dropout import N2N_UNet_dropout
from .logger import Logger
import time
import sys
from tqdm import tqdm
from torchvision import transforms
from core.median_filter import apply_median_filter_gpu_simple

torch.backends.cudnn.benchmark=True

def select_data_type(_tr_data_dir, _te_data_dir, x_avg_num, y_avg_num,args):
    if 'SEM_semi' in args.data_name:
        print(_tr_data_dir)
        print(_te_data_dir)
        tr_data_loader = TrdataLoader(_tr_data_dir, args)
        tr_data_loader = DataLoader(tr_data_loader, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        te_data_loader = TedataLoader_FullImage(_te_data_dir, args)
        te_data_loader = DataLoader(te_data_loader, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    else :
        raise NotImplementedError("Not implemented data type", args.data_type, args.data_name)
    return tr_data_loader, te_data_loader
class Train_FBI(object):
    def __init__(self,_tr_data_dir=None, _te_data_dir=None, _save_file_name = None, _args = None):
        
        self.args = _args
        
        if self.args.pge_weight_dir != None:
            self.pge_weight_dir = './weights/' + self.args.pge_weight_dir
        self.tr_data_loader, self.te_data_loader = select_data_type(_tr_data_dir, _te_data_dir, \
                                                                    self.args.x_f_num, self.args.y_f_num,self.args)

        self.result_psnr_arr = []
        self.result_ssim_arr = []
        self.result_time_arr = []
        self.result_denoised_img_arr = []
        self.result_te_loss_arr = []
        self.result_tr_loss_arr = []
        self.best_psnr = 0
        self.save_file_name = _save_file_name
        self.date = date.today().isoformat()
        self.logger = Logger(self.args.nepochs, len(self.tr_data_loader))
        if "SEM_semi" in self.args.data_name:
            log_folder = "./SEM_semi_log"
        else :
            log_folder = "./log"
        os.makedirs(log_folder,exist_ok=True)
        self.writer = SummaryWriter(log_folder)

        if self.args.loss_function == 'MSE':
            self.loss = mse_bias
            num_output_channel = 1
            self.args.output_type = "linear"
        elif self.args.loss_function == 'N2V': #lower bound
            self.loss = mse_bias
            num_output_channel = 1
            self.args.output_type = "linear"
        elif self.args.loss_function == 'MSE_Affine': # 1st(our case upper bound)
            self.loss = mse_affine
            num_output_channel = 2
            self.args.output_type = "linear"
        elif self.args.loss_function == 'EMSE_Affine':
            
            self.loss = emse_affine
            num_output_channel = 2
            if self.args.with_originalPGparam is False:
                ## load PGE model
                self.pge_model=est_UNet(num_output_channel,depth=3)
                self.pge_model.load_state_dict(torch.load(self.pge_weight_dir))
                self.pge_model=self.pge_model.cuda()
            
                for param in self.pge_model.parameters():
                    param.requires_grad = False
            
            self.args.output_type = "sigmoid"
        else :
            raise NotImplementedError

        if self.args.model_type == 'FC-AIDE':
            # num of parameters :  1098625
            # self.model = FC_AIDE(channel = 1, output_channel = num_output_channel, filters = 64, num_of_layers=10, output_type = self.args.output_type, sigmoid_value = self.args.sigmoid_value)
            self.model = FC_AIDE(channel = 1, output_channel = num_output_channel, filters = 32, num_of_layers=5, output_type = self.args.output_type, sigmoid_value = self.args.sigmoid_value)
        elif self.args.model_type == 'DBSN':
            # original num of parameters :  6612289
            self.model = DBSN_Model(in_ch = 1,
                            out_ch = num_output_channel,
                            mid_ch = 48, #96,
                            blindspot_conv_type = 'Mask',
                            blindspot_conv_bias = True,
                            br1_block_num = 4, #8,
                            br1_blindspot_conv_ks =3,
                            br2_block_num = 4, #8,
                            br2_blindspot_conv_ks = 5,
                            activate_fun = 'Relu')
        elif self.args.model_type == 'UNet':
            self.model = UNet(dim = 1, output_channel = num_output_channel, ngf_factor = 12, depth = 4)
        elif self.args.model_type == 'N2N_UNet':
            self.model = N2N_UNet()
        elif self.args.model_type == 'N2N_UNet_dropout':
            self.model = N2N_UNet_dropout(dropout_rate=self.args.dropout_rate,dropout_type=self.args.dropout_type,
                                          dropout_RPM_mode = self.args.dropout_RPM_mode, rescale_on_eval = self.args.rescale_on_eval)
        elif self.args.model_type == 'RED30':
            self.model = REDNet30(num_layers=self.args.num_layers, num_features=self.args.num_filters)
        elif self.args.model_type == 'NAFNet':
            self.model =  NAFNet(img_channel=1,output_channel= num_output_channel, width=32, middle_blk_num=12,
                      enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2])
        elif self.args.model_type == 'NAFNet_light':
            self.model =  NAFNet(img_channel=1,output_channel= num_output_channel,width=8, middle_blk_num=4,
                      enc_blk_nums=[2,2,4,4], dec_blk_nums=[2,2,2,2]) # 809257
        elif self.args.model_type == 'FBI_Net_dropout':
            self.model = New_model_dropout(channel = 1, output_channel =  num_output_channel, 
                                filters = self.args.num_filters, num_of_layers=self.args.num_layers, case = self.args.model_type, 
                                output_type = self.args.output_type, sigmoid_value = self.args.sigmoid_value,
                                blind = not self.args.non_blind,
                                 dropout_rate=self.args.dropout_rate, rescale_on_eval=self.args.rescale_on_eval, 
                                dropout_type = self.args.dropout_type)
        elif self.args.model_type == 'FBI_Net_tiny':
            self.model = New_model_tiny(channel = 1, output_channel =  num_output_channel,
                                filters = self.args.num_filters, num_of_layers=self.args.num_layers, case = self.args.model_type, 
                                output_type = self.args.output_type, sigmoid_value = self.args.sigmoid_value,
                                blind = not self.args.non_blind,
                                input_dropout = self.args.input_dropout, dropout_rate=self.args.dropout_rate,
                                rescale_on_eval=self.args.rescale_on_eval)
        else:
            self.model = New_model(channel = 1, output_channel =  num_output_channel, 
                                filters = self.args.num_filters, num_of_layers=self.args.num_layers, case = self.args.model_type, 
                                output_type = self.args.output_type, sigmoid_value = self.args.sigmoid_value,
                                blind = not self.args.non_blind,
                                input_dropout = self.args.input_dropout, dropout_rate=self.args.dropout_rate,
                                rescale_on_eval=self.args.rescale_on_eval)
        if self.args.load_ckpt is not None:
            model_weight = torch.load(self.args.load_ckpt)
            self.model.load_state_dict(model_weight)
        self.model = self.model.cuda()
        
        pytorch_total_params = sum([p.numel() for p in self.model.parameters()])
        print ('num of parameters : ', pytorch_total_params)
        if self.args.log_off is False:
            self.args.logger.log({})
        if self.args.batch_test is True:
            z = torch.randn(1,1,256,256).cuda()
            out = self.model(z)
            if self.args.model_type == 'DBSN':
                out, _ = out
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.model_type == 'N2N_UNet' or self.args.model_type == 'RED30':
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                          betas=(0.9, 0.99), eps=1e-08)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, self.args.drop_epoch, gamma=self.args.drop_rate)
        
    def save_model(self,num_epoch = None):
        os.makedirs('./weights',exist_ok=True)
        if num_epoch is not None:
            
            torch.save(self.model.state_dict(), f"./weights/{self.save_file_name}_{num_epoch:02d}.w")
        else :
            torch.save(self.model.state_dict(), f"./weights/{self.save_file_name}.w")
            
        
        return
    
    def get_X_hat(self, Z, output):

        X_hat = output[:,:1] * Z + output[:,1:]
            
        return X_hat
    def update_log(self,epoch,output):
        # args = self.args
        mean_tr_loss, mean_te_loss, mean_psnr, mean_ssim = output
        if self.args.log_off is False:
            self.args.logger.log({
                    'train/mean_loss'   : mean_tr_loss,
                    'test/mean_loss'    : mean_te_loss,
                    'mean_psnr'         : mean_psnr,
                    'mean_ssim'         : mean_ssim,
            })
        

    def eval(self,epoch):
        """Evaluates denoiser on validation set."""
        psnr_arr = []
        ssim_arr = []
        loss_arr = []
        time_arr = []
        denoised_img_arr = []
        if self.args.eval_off is False:
            self.model.eval()
        else:
            print("eval off on the model")
        # # next prob-BSN
        with torch.no_grad():

            for batch_idx, (source, target) in tqdm(enumerate(self.te_data_loader)):

                start = time.time()
                    
                source = source.cuda()
                target = target.cuda()
                if self.args.loss_function =='EMSE_Affine':
                    if self.args.with_originalPGparam is True:
                        original_alpha=torch.tensor(self.args.alpha).unsqueeze(0).cuda()
                        original_sigma=torch.tensor(self.args.beta).unsqueeze(0).cuda()
                    else :
                        est_param=self.pge_model(source)
                        original_alpha=torch.mean(est_param[:,0])
                        original_sigma=torch.mean(est_param[:,1])
                    if self.args.batch_test is True:
                        print('original_alpha : ',original_alpha)
                        print('original_sigma : ',original_sigma)
                    transformed=gat(source,original_sigma,original_alpha,0)
                    transformed, transformed_sigma, min_t, max_t= normalize_after_gat_torch(transformed)
                    
                    transformed_target = torch.cat([transformed, transformed_sigma], dim = 1)

                    output = self.model(transformed)
    
                    loss = self.loss(output, transformed_target)
        
                else:
                    # Denoise image
                    if self.args.model_type == 'DBSN':
                        output, _ = self.model(source)
                        loss = self.loss(output, target)
                    else:
                        if self.args.model_type == 'N2N_UNet' or self.args.model_type == 'RED30':
                            # scale to -0.5 ~ 0.5
                            source = source.float() - 0.5
                        if epoch == 0:
                            median_filtered_img = apply_median_filter_gpu_simple(source.clone())[0][0]
                        output = self.model(source)
                        loss = self.loss(output, target)

                loss = loss.cpu().numpy()
                
                if self.args.loss_function == 'MSE':
                    X = target.cpu().numpy()
                    X_hat = output.cpu().numpy()
                    X_hat = np.clip(X_hat, 0, 1)
                    if epoch == 0:
                        median_filtered_img = np.clip(median_filtered_img, 0, 1)
                    
                elif self.args.loss_function == 'MSE_Affine':
                    
                    Z = target[:,:1]
                    X = target[:,1:].cpu().numpy()
                    X_hat = self.get_X_hat(Z,output).cpu().numpy()
                    X_hat = np.clip(X_hat, 0, 1)
                    
                elif  self.args.loss_function == 'N2V':
                    X = target[:,1:].cpu().numpy()
                    X_hat = output.cpu().numpy()
                    X_hat = np.clip(X_hat, 0, 1)
                    
                
                elif self.args.loss_function == 'EMSE_Affine':
                    
                    transformed_Z = transformed_target[:,:1]
                    X = target.cpu().numpy()
                    X_hat = self.get_X_hat(transformed_Z,output).cpu().numpy()
                    
                    transformed=transformed.cpu().numpy()
                    original_sigma=original_sigma.cpu().numpy()
                    original_alpha=original_alpha.cpu().numpy()
                    min_t=min_t.cpu().numpy()
                    max_t=max_t.cpu().numpy()
                    X_hat =X_hat*(max_t-min_t)+min_t
                    X_hat=np.clip(inverse_gat(X_hat,original_sigma,original_alpha,0,method='closed_form'), 0, 1)
                

                inference_time = time.time()-start
                
                loss_arr.append(loss)
                psnr_arr.append(get_PSNR(X[0], X_hat[0]))
                ssim_arr.append(get_SSIM(X[0], X_hat[0]))
                if epoch ==0:
                    psnr_median_filter = get_PSNR(X[0], median_filtered_img)
                    ssim_median_filter = get_SSIM(X[0], median_filtered_img)
                denoised_img_arr.append(X_hat[0].reshape(X_hat.shape[2],X_hat.shape[3]))
                if self.args.log_off is False:
                    name_str = f'batch_{batch_idx}'
                    image = wandb.Image(denoised_img_arr[-1], caption = f'EPOCH : {epoch} Batch : {batch_idx}\nPSNR: {psnr_arr[-1]:.4f}, SSIM: {ssim_arr[-1]:.4f}')
                    if epoch == 0:
                        noisy_image = wandb.Image(source[0].cpu().numpy(), caption = f'noisy_img_{name_str}')
                        median_filtered_image = wandb.Image(median_filtered_img, caption = f'median_filtered_img\nPSNR : {psnr_median_filter:.4f}, SSIM : {ssim_median_filter:.4f}')
                        self.args.logger.log({f"eval/noisy_img_{name_str}" : noisy_image})
                        self.args.logger.log({f"eval/median_filtered_img" : median_filtered_image})
                    self.args.logger.log({f"eval/denoised_img_{name_str}" : image},step=epoch)
                    
                time_arr.append(inference_time)

        mean_loss = np.mean(loss_arr)
        mean_psnr = np.mean(psnr_arr)
        mean_ssim = np.mean(ssim_arr)
        mean_time = np.mean(time_arr)
        if self.args.log_off is False:
            self.args.logger.log({f"eval/loss" : mean_loss},step=epoch)
            self.args.logger.log({f"eval/psnr" : mean_psnr},step=epoch)
            self.args.logger.log({f"eval/ssim" : mean_ssim},step=epoch) 
        if self.best_psnr <= mean_psnr:
            self.best_psnr = mean_psnr
            self.result_denoised_img_arr = denoised_img_arr.copy()
        
            
        return mean_loss, mean_psnr, mean_ssim, mean_time
    
    def _on_epoch_end(self, epoch, mean_tr_loss):
        """Tracks and saves starts after each epoch."""
        print(f"\n==== start evaluation at {epoch}/{self.args.nepochs} =====  : ")
        mean_te_loss, mean_psnr, mean_ssim, mean_time = self.eval(epoch)

        self.result_psnr_arr.append(mean_psnr)
        self.result_ssim_arr.append(mean_ssim)
        self.result_time_arr.append(mean_time)
        self.result_te_loss_arr.append(mean_te_loss)
        self.result_tr_loss_arr.append(mean_tr_loss)
        if self.args.log_off is False:
            sio.savemat('./result_data/'+self.save_file_name + '_result',{'tr_loss_arr':self.result_tr_loss_arr, 'te_loss_arr':self.result_te_loss_arr,'psnr_arr':self.result_psnr_arr, 'ssim_arr':self.result_ssim_arr,'time_arr':self.result_time_arr, 'denoised_img':self.result_denoised_img_arr})

        print ('Epoch : ', epoch, ' Tr loss : ', round(mean_tr_loss,4), ' Te loss : ', round(mean_te_loss,4),
             ' PSNR : ', round(mean_psnr,4), ' SSIM : ', round(mean_ssim,4),' Best PSNR : ', round(self.best_psnr,4)) 
        if self.args.batch_test is True:
            print("==== batch_test end ====")
            sys.exit(0)
        if self.args.log_off is False:
            self.update_log(epoch,[mean_tr_loss, mean_te_loss, mean_psnr, mean_ssim])
        return mean_psnr,mean_ssim
    def train(self):
        """Trains denoiser on training set."""
        self.model.train()
        for epoch in range(self.args.nepochs):

            tr_loss = []

            for batch_idx, (source, target) in enumerate(self.tr_data_loader):

                self.optim.zero_grad()

                source = source.cuda()
                target = target.cuda()
                
                # Denoise
                if self.args.apply_RPM_before_model:
                    source = apply_RPM(source, self.args.RPM_p, RPM_type=self.args.RPM_type,
                                            RPM_grid_size=self.args.RPM_grid_size,
                                            RPM_masking_value=self.args.RPM_masking_value)
                if self.args.loss_function =='EMSE_Affine':
                    if self.args.with_originalPGparam is True:
                        original_alpha = self.args.alpha
                        original_sigma = self.args.beta
                    else :
                        est_param=self.pge_model(source)
                        original_alpha=torch.mean(est_param[:,0])
                        original_sigma=torch.mean(est_param[:,1])
                    if self.args.batch_test is True:
                        print('original_alpha : ',original_alpha)
                        print('original_sigma : ',original_sigma)
                        
                    
                    transformed=gat(source,original_sigma,original_alpha,0)
                    transformed, transformed_sigma, min_t, max_t= normalize_after_gat_torch(transformed)
                    
                    target = torch.cat([transformed, transformed_sigma], dim = 1)

                    output = self.model(transformed)
                else:
                    # Denoise image
                    if self.args.model_type == 'DBSN':
                        output, _ = self.model(source)
                    else:
                        if self.args.model_type == 'N2N_UNet' or self.args.model_type == 'RED30':
                            # scale to -0.5 ~ 0.5
                            source = source - 0.5
                        if self.args.train_time_RPM:
                            # mask = np.random.binomial(1, self.args.RPM_p, size=source.shape).astype(bool)
                            mask = torch.bernoulli(torch.full_like(source, self.args.RPM_p)).bool()
                            source[mask] = self.args.RPM_masking_value
                            if self.args.rescale_after_RPM:
                                source = source * (1/(1-self.args.RPM_p))
                        output = self.model(source)
                    
                loss = self.loss(output, target)
                    
                loss.backward()
                self.optim.step()
                # if self.args.log_off is False:
                self.logger.log(losses = {'loss': loss}, lr = self.optim.param_groups[0]['lr'])

                tr_loss.append(loss.detach().cpu().numpy())
                if self.args.batch_test is True and batch_idx > 10:
                    break
            mean_tr_loss = np.mean(tr_loss)
            
            mean_psnr,mean_ssim = self._on_epoch_end(epoch+1, mean_tr_loss)   
            if self.args.save_whole_model is True:
                self.save_model(epoch+1)
            elif ((epoch+1) % 10) == 0:
                self.save_model(epoch+1)
            elif self.args.nepochs == epoch +1:
                self.save_model()
                
            self.scheduler.step()
            if self.args.batch_test is True:
                break
            if self.args.test is True and epoch > 2:
                print("==== 3 epoch test end ====")
                sys.exit(0)
        self.writer.close()
        
        return mean_psnr,mean_ssim




