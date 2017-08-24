#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -p debug
#SBATCH --mem=5000

cd ..

netG=resnet_9blocks
#netG=unet_256
nonlinear=sigmoid
lambda_A=100
lambda_TV=0.00001

name=disentangled_${netG}_${nonlinear}_A${lambda_A}_TV${lambda_TV}
model=disentangled

dataroot=/home-4/xyang35@umd.edu/data/xyang/Haze/IVCDehazingDataset/results/
checkpoints_dir=/home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/checkpoints
results_dir=/home-4/xyang35@umd.edu/data/xyang/Haze/IVCDehazingDataset/results/

python test.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --results_dir $results_dir \
    --name $name --model $model --which_model_depth aod --which_model_netG $netG \
    --dataset_mode depth --display_id 0 --serial_batches --phase test \
    --non_linearity $nonlinear  --pooling --no_dropout --gpu_ids -1
#    --non_linearity $nonlinear  --pooling --norm batch 

cd $results_dir
zip ${name}_results.zip $name -r
