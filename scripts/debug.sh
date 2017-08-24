#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p bw-gpu
#SBATCH --mem=10000
#SBATCH --gres=gpu:2

cd ..

nonlinear=BReLU
name=debug_pool_${nonlinear}
#name=debug_${nonlinear}
model=debug

dataroot=/home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/debug
checkpoints_dir=/home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/checkpoints
results_dir=/home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/results/

python train.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --name $name --model $model --which_model_depth aod --no_flip --non_linearity $nonlinear \
    --niter 20  --niter_decay 20 --grad_clip 0.1 --lambda_perceptual 0. --lr 0.00001 --lambda_TV 0. --lambda_A 1 --pooling\
    --gpu_ids 0,1 --batchSize 8 --display_id 0  --dataset_mode unaligned 


python test.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --results_dir $results_dir \
    --name $name --model $model --which_model_depth aod  --no_flip --non_linearity $nonlinear --pooling\
    --dataset_mode unaligned --display_id 0 --serial_batches --phase test

# pack the results
cd $checkpoints_dir/$name
zip ${name}_checkpoints.zip web/ -r

cd $results_dir
zip ${name}_results.zip $name -r
