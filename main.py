from tkinter import N
from core.train_fbi import Train_FBI
from core.train_pge import Train_PGE
from arguments import get_args
import os,sys
import torch
import numpy as np
import random
import wandb
print("torch version : ",torch.__version__)
args = get_args()

# control the randomness
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

if args.log_off is True:
    args.logger = {}
else :
    model_info = f"{args.model_type}_{args.x_f_num}-{args.y_f_num}"
    details = f"{args.loss_function}_epoch{args.nepochs}_lr{args.lr}_bs{args.batch_size}"
    run_id = model_info + "_" + details
    if args.non_blind is True:
        run_id += '_nonblind_new'
    if args.apply_RPM is True:
        run_id += f"RPM_p{args.RPM_p}_{args.RPM_type}"
        if args.RPM_type == 'grid':
            run_id += f"_gridsize_{args.RPM_grid_size}"
        if args.rescale_after_RPM :
            run_id += f"_rescale_after_RPM"
    if 'dropout' in args.model_type:
        run_id += f"_dropout_{args.dropout_rate}_{args.dropout_type}"
    if args.RPM_masking_value != 0:
        run_id += f"_masking{args.RPM_masking_value}"
    if args.input_dropout is True:
        run_id += f"_input_dropout_{args.dropout_rate}"
        if args.dropout_RPM_mode:
            run_id += f"_RPM_mode"
        if args.rescale_on_eval:
            run_id += f"_rescale_on_eval"
    if args.train_time_RPM:
        run_id += f"_train_time_RPM"
        if args.rescale_after_RPM :
            run_id += f"_rescale_after_RPM"
    

    project_id = 'ReNIn'
    args.logger = wandb.init(anonymous="allow", project=project_id)

    wandb.run.name = run_id

os.makedirs("./result_data", exist_ok=True)
if __name__ == '__main__':
    if args.test is True:
        args.nepochs = 2
    save_file_name = f"{args.date}_{args.model_type}_{args.loss_function}_RN2N_{args.x_f_num}-{args.y_f_num}_{args.data_name}_ep{args.nepochs}"
    if args.test is True:
        save_file_name += '_2epochtest'
    #semiconductor SEM image
    if args.data_name == 'SEM_semi':
        tr_data_dir = f'./train.hdf5'
        te_data_dir = f'./test.hdf5'
    else:
        raise NotImplementedError
    print ('tr data dir : ', tr_data_dir)
    print ('te data dir : ', te_data_dir)
                       
    if args.non_blind is True:
        save_file_name += '_nonblind_new'
    if args.apply_RPM is True:
        save_file_name += f"_RPM_p{args.RPM_p}_type_{args.RPM_type}"
        if args.RPM_type == 'grid':
            save_file_name += f"_gridsize_{args.RPM_grid_size}"
        if args.rescale_after_RPM :
            save_file_name += f"_rescale_after_RPM"
        if args.train_time_RPM:
            save_file_name += f"_train_time_RPM"
    
    if args.RPM_masking_value != 0:
        save_file_name += f"_masking{args.RPM_masking_value}"
    if args.input_dropout is True:
        save_file_name += f"_input_dropout_{args.dropout_rate}"
        if args.dropout_RPM_mode:
            save_file_name += f"_RPM_mode"
    if args.rescale_on_eval:
        save_file_name += f"_rescale_on_eval"
        if args.rescale_after_RPM :
            save_file_name += f"_rescale_after_RPM"
    # model specific name
    if args.model_type == 'FC-AIDE':
        save_file_name += '_layers_x' + str(10) + '_filters_x' + str(64)
    elif args.model_type == 'DBSN':
        save_file_name = ''
    elif args.model_type == 'PGE_Net':
        save_file_name += f"_cropsize_{args.crop_size}_vst_{args.vst_version}"
    elif args.model_type == 'FBI_Net':
        save_file_name += '_layers_x' + str(args.num_layers) + '_filters_x' + str(args.num_filters)+ '_cropsize_' + str(args.crop_size)
    elif 'dropout' in args.model_type:
        save_file_name += f"_dropout_{args.dropout_rate}_{args.dropout_type}"
    if args.log_off is False:
        args.logger.config.update({'Network' : args.model_type})
        args.logger.config.update({'save_filename' : save_file_name})

        args.logger.config.update(args)
    save_file_name += f"_testidx_{args.testidx}"
    if args.load_ckpt is not None:
        info = args.load_ckpt.split("SEM_semi_")[1]
        info = info.split("_layers")[0]
        save_file_name += f"_from_{info}"
    print ('save_file_name : ', save_file_name)
    
    
    if args.model_type != 'PGE_Net':
        train = Train_FBI(_tr_data_dir=tr_data_dir, _te_data_dir=te_data_dir, _save_file_name = save_file_name,  _args = args)
    else:
        train = Train_PGE(_tr_data_dir=tr_data_dir, _te_data_dir=te_data_dir, _save_file_name = save_file_name,  _args = args)
    train.train()
    
    print ('Finsh training - save_file_name : ', save_file_name)
