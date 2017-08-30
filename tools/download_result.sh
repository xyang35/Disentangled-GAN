#!/bin/bash

name=$1

checkpoints_dir=/home-4/xyang35@umd.edu/work/xyang/GAN/Haze/D-HAZY/checkpoints
results_dir=/home-4/xyang35@umd.edu/work/xyang/GAN/Haze/D-HAZY/results

mkdir $name

cd $name
#scp xyang35@umd.edu@gateway2.marcc.jhu.edu:${checkpoints_dir}/$name/${name}_checkpoints.zip ./
scp xyang35@umd.edu@gateway.marcc.jhu.edu:${results_dir}/${name}_results.zip /home/xyang/UTS/Data/Haze/D-HAZY/NYU/results/

cd /home/xyang/UTS/Data/Haze/D-HAZY/NYU/results/
#unzip ${name}_checkpoints.zip && rm ${name}_checkpoints.zip
unzip ${name}_results.zip && rm ${name}_results.zip
