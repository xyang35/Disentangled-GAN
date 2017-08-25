#!/bin/bash

cd ..

name=cyclegan
model=cycle_gan

dataroot=/home/xyang/UTS/Data/Haze/D-HAZY/NYU
checkpoints_dir=/home/xyang/UTS/Data/Haze/D-HAZY/NYU/checkpoints
results_dir=/home/xyang/UTS/Data/Haze/D-HAZY/NYU/results

python train.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --name $name --model $model --display_id 0 \
    --pool_size 50 --niter 50  --niter_decay 50 \
    --no_dropout --gpu_ids 0,1 --batchSize 8

python test.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --results_dir $results_dir \
    --name $name --model $model --display_id 0\
    --phase test --no_dropout --serial_batches --phase test --how_many 500

