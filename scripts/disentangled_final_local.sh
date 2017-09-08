#!/bin/bash

# exit if not suceed
set -e

cd ..

#nonlinear=sigmoid
lambda_A=40
lambda_TV=0
lr=0.0002
filtering=none
gpu_ids=$2
id=$1

name=disentangled_final_A${lambda_A}_TV${lambda_TV}_lr${lr}_${filtering}_id${id}
model=disentangled_final

dataroot=/home/xyang/UTS/Data/Haze/D-HAZY/final
checkpoints_dir=/home/xyang/UTS/Data/Haze/D-HAZY/final/checkpoints
results_dir=/home/xyang/UTS/Data/Haze/D-HAZY/final/results

python train.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --name $name --model $model --which_model_depth resnet_depth --which_model_netG resnet_9blocks --which_model_netD multi \
    --lambda_A $lambda_A --lambda_TV $lambda_TV \
    --niter 50  --niter_decay 50  --pool_size 50 --no_dropout --lr $lr --resize_or_crop scale_width --loadSize 256 \
    --gpu_ids $gpu_ids --batchSize 8 --display_id 0  --dataset_mode depth --depth_reverse

python test.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --results_dir ${results_dir} \
    --name $name --model $model --which_model_depth resnet_depth --which_model_netG resnet_9blocks --which_model_netD multi \
    --no_dropout --depth_reverse --resize_or_crop scale_width  --loadSize 256 \
    --gpu_ids $gpu_ids --dataset_mode depth --display_id 0 --phase test --how_many 500


# evaluation
echo "Evaluation ..."

cd tools
matlab <<< "name = '$name'; evaluation;"
