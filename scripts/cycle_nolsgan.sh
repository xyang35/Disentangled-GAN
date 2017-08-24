#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH --mem=10000
#SBATCH --gres=gpu:1

cd ..

name=cycle_nolsgan_nyu
model=cycle_gan

dataroot=/home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/NYU
checkpoints_dir=/home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/checkpoints
results_dir=/home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/results/

#python train.py --dataroot $dataroot \
#    --checkpoints_dir $checkpoints_dir \
#    --name $name --model $model --display_id 0 \
#    --pool_size 50 --niter 50  --niter_decay 50 \
#    --no_dropout --no_lsgan

#python test.py --dataroot $dataroot \
#    --checkpoints_dir $checkpoints_dir \
#    --results_dir $results_dir \
#    --name $name --model $model --display_id 0\
#    --phase test --no_dropout 

# pack the results
cd $checkpoints_dir/$name
zip ${name}_checkpoints.zip web/ -r

cd $results_dir
zip ${name}_results.zip $name -r
