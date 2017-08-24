#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p bw-gpu
#SBATCH --mem=10000
#SBATCH --gres=gpu:2

cd ..

name=recongan_resnet_nyu
model=recon

dataroot=/home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/NYU
checkpoints_dir=/home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/checkpoints
results_dir=/home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/results/

python train.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --name $name --model $model --no_dropout\
    --pool_size 50 --niter 100  --niter_decay 50 --lambda_A 100 \
    --gpu_ids 0,1 --batchSize 8 --display_id 0  --dataset_mode depth --depth_reverse    # reverse depth as transition map

python test.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --results_dir $results_dir \
    --name $name --model $model --no_dropout \
    --phase test --which_epoch 100  --dataset_mode depth --display_id 0 --serial_batches --depth_reverse

python test.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --results_dir $results_dir \
    --name $name --model $model --no_dropout \
    --phase test --dataset_mode depth --display_id 0 --serial_batches --depth_reverse

# pack the results
cd $checkpoints_dir/$name
zip ${name}_checkpoints.zip web/ -r

cd $results_dir
zip ${name}_results.zip $name -r
