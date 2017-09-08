#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH --mem=10000
#SBATCH --gres=gpu:2

# exit if not suceed
set -e

cd ..

nonlinear=sigmoid
lambda_A=40
lambda_TV=0.1
lr=0.0002
filtering=guided
gpu_ids=$2
id=$1

name=disentangled_${nonlinear}_A${lambda_A}_TV${lambda_TV}_lr${lr}_${filtering}_id${id}
model=disentangled_final

dataroot=/home-4/xyang35@umd.edu/work/xyang/GAN/Haze/D-HAZY/NYU
checkpoints_dir=/home-4/xyang35@umd.edu/work/xyang/GAN/Haze/D-HAZY/checkpoints
results_dir=/home-4/xyang35@umd.edu/work/xyang/GAN/Haze/D-HAZY/results/

python train.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --name $name --model $model --which_model_depth unet_256 --which_model_netG resnet_9blocks --which_model_netD multi \
    --non_linearity $nonlinear --lambda_A $lambda_A --lambda_TV $lambda_TV --filtering $filtering \
    --niter 50  --niter_decay 50  --pool_size 50 --no_dropout --lr $lr \
    --gpu_ids $gpu_ids --batchSize 8 --display_id 0  --dataset_mode depth --depth_reverse

python test.py --dataroot $dataroot \
    --checkpoints_dir $checkpoints_dir \
    --results_dir ${results_dir} \
    --name $name --model $model --which_model_depth unet_256 --which_model_netG resnet_9blocks --which_model_netD multi \
    --non_linearity $nonlinear  --no_dropout --depth_reverse --filtering $filtering \
    --dataset_mode depth --display_id 0 --serial_batches --phase test --how_many 500

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
