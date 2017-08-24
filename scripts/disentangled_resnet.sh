#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p bw-gpu
#SBATCH --mem=10000
#SBATCH --gres=gpu:2

cd ..

netG=resnet_9blocks
nonlinear=BReLU
lambda_A=100
lambda_TV=0.00001

name=disentangled_${netG}_${nonlinear}_A${lambda_A}_TV${lambda_TV}
model=disentangled

dataroot=/home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/NYU
checkpoints_dir=/home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/checkpoints
results_dir=/home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/results/

python train.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --name $name --model $model --which_model_depth aod --which_model_netG $netG \
    --non_linearity $nonlinear --lambda_A $lambda_A --lambda_TV $lambda_TV --pooling \
    --niter 50  --niter_decay 50  --pool_size 50 --no_dropout \
    --gpu_ids 0,1 --batchSize 8 --display_id 0  --dataset_mode depth --depth_reverse

python test.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --results_dir $results_dir \
    --name $name --model $model --which_model_depth aod --which_model_netG $netG \
    --non_linearity $nonlinear  --pooling --no_dropout --depth_reverse \
    --dataset_mode depth --display_id 0 --serial_batches --phase test --which_epoch 50

python test.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --results_dir $results_dir \
    --name $name --model $model --which_model_depth aod --which_model_netG $netG \
    --non_linearity $nonlinear  --pooling --no_dropout --depth_reverse \
    --dataset_mode depth --display_id 0 --serial_batches --phase test

# pack the results
cd $checkpoints_dir/$name
zip ${name}_checkpoints.zip web/ -r

cd $results_dir
zip ${name}_results.zip $name -r
