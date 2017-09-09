#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p bw-gpu
#SBATCH --mem=10000
#SBATCH --gres=gpu:1

# exit if not suceed
set -e

cd ..

#nonlinear=sigmoid
lambda_A=40
lambda_TV=0
lr=0.0002
filtering=none
id=$1

name=disentangled_final_A${lambda_A}_TV${lambda_TV}_lr${lr}_${filtering}_id${id}
model=disentangled_final

dataroot=/home-4/xyang35@umd.edu/work/xyang/GAN/Haze/D-HAZY/final
checkpoints_dir=/home-4/xyang35@umd.edu/work/xyang/GAN/Haze/D-HAZY/final/checkpoints
results_dir=/home-4/xyang35@umd.edu/work/xyang/GAN/Haze/D-HAZY/final/results/

python train.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --name $name --model $model --which_model_depth resnet9_depth --which_model_netG resnet_9blocks --which_model_netD multi \
    --lambda_A $lambda_A --lambda_TV $lambda_TV \
    --niter 50  --niter_decay 50  --pool_size 50 --no_dropout --lr $lr --resize_or_crop disentangled \
    --batchSize 8 --display_id 0  --dataset_mode depth --depth_reverse

python test.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --results_dir ${results_dir} \
    --name $name --model $model --which_model_depth resnet9_depth --which_model_netG resnet_9blocks --which_model_netD multi \
    --no_dropout --depth_reverse --resize_or_crop disentangled \
    --dataset_mode depth --display_id 0 --serial_batches --phase test --how_many 1500

# pack the results
#cd $checkpoints_dir/$name
#zip ${name}_checkpoints.zip web/ -r

#cd $results_dir
#zip ${name}_results.zip $name -r

# evaluation
echo "Evaluation ..."

cd tools
module load matlab
matlab <<< "name = '$name'; evaluation_server;"
