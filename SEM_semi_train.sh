#! /bin/bash
DATE=`date "+%y%m%d"`
# Synthetic noise datasets
DATA_TYPE='Grayscale'
DATA_NAME='SEM_semi'
GPU_NUM=$1
X_F_NUM=$2
Y_F_NUM=$3
LOSS_FUNCTION=$4 # 'MSE_with_l1norm_on_gradient'
ARCH=$5
BATCH_SIZE=$6
shift 6
OPTION=$@
echo "GPU_NUM :"${GPU_NUM}
echo "OPTION : "${OPTION}
# echo "X_F_NUM : F"$X_F_NUM ", Y_F_NUM : F"${Y_F_NUM}
echo "BATCH Size : "${BATCH_SIZE}
# ALPHA == 0, BETA == 0 : Mixture Noise
ALPHA=0.0
BETA=0.0 # == sigma of Poisson Gaussian noise

echo $DATE
echo "=== Train FBI with "$LOSS_FUNCTION " " $X_F_NUM"<->"$Y_F_NUM"==="
CUDA_VISIBLE_DEVICES=$GPU_NUM python main.py $OPTION --integrate-all-set --individual-noisy-input \
    --x-f-num $X_F_NUM --y-f-num $Y_F_NUM \
    --date $DATE --seed 0 --noise-type 'Poisson-Gaussian' --loss-function $LOSS_FUNCTION \
    --model-type $ARCH --data-type $DATA_TYPE --data-name $DATA_NAME \
    --alpha $ALPHA --beta $BETA --batch-size $BATCH_SIZE --lr 0.001 --num-layers 17 \
    --num-filters 64 --crop-size 256 
