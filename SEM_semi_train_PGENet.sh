#!/bin/bash


DATE=`date "+%y%m%d"`
DATA_TYPE='Grayscale'
DATA_NAME='SEM_semi'

BATCH_SIZE=1
GPU_NUM=`expr $1 % 4`
X_F_NUM=$2
Y_F_NUM=$3
shift 3
OPTION=$@
echo "GPU_NUM:"$GPU_NUM
echo "X_F_NUM:"$X_F_NUM " Y_F_NUM:"$Y_F_NUM
ALPHA=0.0
BETA=0.0

echo $pge_net_weight_path "not exist, Train PGE"
CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py $OPTION --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' \
    --integrate-all-set --individual-noisy-input --x-f-num $X_F_NUM --y-f-num $Y_F_NUM --nepochs 50 \
    --loss-function 'Noise_est' --model-type 'PGE_Net' --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --batch-size $BATCH_SIZE --lr 0.0001 --crop-size 256 --testidx 4  

