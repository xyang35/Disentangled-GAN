#!/bin/bash

name=$1

checkpoints_dir=/home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/checkpoints
results_dir=/home-4/xyang35@umd.edu/data/xyang/Haze/D-HAZY/results

mkdir $name

cd $name
scp xyang35@umd.edu@gateway2.marcc.jhu.edu:${checkpoints_dir}/$name/${name}_checkpoints.zip ./
#scp xyang35@umd.edu@gateway.marcc.jhu.edu:${results_dir}/${name}_results.zip ./

unzip ${name}_checkpoints.zip && rm ${name}_checkpoints.zip
#unzip ${name}_results.zip && rm ${name}_results.zip
