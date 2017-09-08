#!/bin/bash

# exit if not suceed
set -e

cd ..

nonlinear=sigmoid
lambda_A=10
lambda_TV=0
lambda_extra=1
lr=0.0002
gpu_ids=$2
id=$1

#name=disentangled_${netG}_${nonlinear}_A${lambda_A}_TV${lambda_TV}
name=disentangled_extra_A${lambda_A}_TV${lambda_TV}_lr${lr}_extra${lambda_extra}_id${id}
model=disentangled_extra

dataroot=/home/xyang/UTS/Data/Haze/D-HAZY/NYU
checkpoints_dir=/home/xyang/UTS/Data/Haze/D-HAZY/NYU/checkpoints
results_dir=/home/xyang/UTS/Data/Haze/D-HAZY/NYU/results

python train.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --name $name --model $model --which_model_depth unet_256 --which_model_netG resnet_9blocks --which_model_netD multi \
    --non_linearity $nonlinear --lambda_A $lambda_A --lambda_TV $lambda_TV --lambda_extra $lambda_extra \
    --niter 50  --niter_decay 50  --pool_size 50 --no_dropout --lr $lr \
    --gpu_ids $gpu_ids --batchSize 8 --display_id 0  --dataset_mode depth --depth_reverse

python test.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --results_dir ${results_dir} \
    --name $name --model $model --which_model_depth unet_256 --which_model_netG resnet_9blocks --which_model_netD multi \
    --non_linearity $nonlinear  --no_dropout --depth_reverse \
    --dataset_mode depth --display_id 0 --serial_batches --phase test --how_many 500


# evaluation
echo "Evaluation ..."

cd tools
matlab <<< "name = '$name'; evaluation;"
