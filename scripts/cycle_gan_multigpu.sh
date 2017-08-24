#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p bw-gpu
#SBATCH --mem=10000
#SBATCH --gres=gpu:2

cd ..

name=cyclegan2
model=cycle_gan

python train.py --dataroot /home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/NYU \
    --checkpoints_dir /home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/checkpoints \
    --name $name --model $model --display_id 0 \
    --pool_size 50 --niter 50  --niter_decay 50 \
    --no_dropout --gpu_ids 0,1 --batchSize 8

python test.py --dataroot /home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/NYU \
    --checkpoints_dir /home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/checkpoints \
    --results_dir /home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/results/ \
    --name $name --model $model --display_id 0\
    --phase test --no_dropout 

python test.py --dataroot /home-4/xyang35@umd.edu/data/xyang/Haze/IVCDehazingDataset/results/ \
    --checkpoints_dir /home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/checkpoints \
    --results_dir /home-4/xyang35@umd.edu/data/xyang/Haze/IVCDehazingDataset/results/ \
    --name $name --model $model --display_id 0\
    --phase test --no_dropout 
