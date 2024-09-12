#!/bin/bash
BATCH_SIZE=1
GPU_NUM=0
##########################################################################
# Noise2Noise 
# ./SEM_semi_train.sh $GPU_NUM F01 F01 MSE FBI_Net $BATCH_SIZE --nepochs 10 --testidx 4 
#####################################
# RelaxNoise2Noise
# ./SEM_semi_train.sh $GPU_NUM F01 F08 MSE FBI_Net $BATCH_SIZE --nepochs 10 --testidx 4
#####################################
# supervised learning
# ./SEM_semi_train.sh $GPU_NUM F01 F64 MSE FBI_Net $BATCH_SIZE --nepochs 10 --testidx 4
##########################################################################
# Noise2Noise + Input dropout
# ./SEM_semi_train.sh $GPU_NUM F01 F01 MSE FBI_Net $BATCH_SIZE --nepochs 10 --testidx 4 \
#     --apply-RPM --RPM-p 0.6 --RPM-type random 
#####################################
# ReNIn (Relaxed NoiseNoise + Input dropout)
# ./SEM_semi_train.sh $GPU_NUM F01 F08 MSE FBI_Net $BATCH_SIZE --nepochs 10 --testidx 4 \
#     --apply-RPM --RPM-p 0.6 --RPM-type random 
#####################################
# Supervised learning + Input dropout
# ./SEM_semi_train.sh $GPU_NUM F01 F64 MSE FBI_Net $BATCH_SIZE --nepochs 10 --testidx 4 \
#     --apply-RPM --RPM-p 0.6 --RPM-type random 

##########################################################################
# other baselines
##########################################################################
# bm3d
# usage : bm3d $input_img $output_img $sigma
# bm3d dataset/F01_8.png  1/F01_8.png tmp.png 140
#####################################
# Noise2Void
# ./SEM_semi_train.sh $GPU_NUM F01 F64 N2V FBI_Net 1 --nepochs 10 --testidx 4 
#####################################
# FBI-denoiser
# ./SEM_semi_train_PGENet.sh $GPU_NUM F01 F64 PGE FBI_Net 
## set your pre-trained PGE weight
# PGE_WEIGHT="240214_PGE_Net_Noise_est_RN2N_F01-F64_SEM_semi_ep50_cropsize_256_vst_MSEtestidx_4.w"
# ./SEM_semi_train.sh $GPU_NUM F01 F64 EMSE_Affine FBI_Net $BATCH_SIZE --testidx 4 \
#     --pge-weight-dir $PGE_WEIGHT



