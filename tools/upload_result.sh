#!/bin/bash

#name='disentangled_shuffle_resnet_9blocks_sigmoid_A100_TV1_lr0.0002'
#name='disentangled_resnet_9blocks_sigmoid_A10_TV1_lr0.0002'
name=$1

mkdir demo/app/static/$name

cp /home/xyang/UTS/Data/Haze/D-HAZY/NYU/results/${name}/test_latest/images demo/app/static/${name} -r
#cp DCP/${name} demo/app/static/DCP -r
#cp DehazeNet/${name} demo/app/static/DehazeNet -r

python check_eval.py $name
mv Check_evaluation_${name}.html demo/app/templates/

python visualize.py $name
mv Dehaze_${name}.html demo/app/templates/
mv Transmission_${name}.html demo/app/templates/
