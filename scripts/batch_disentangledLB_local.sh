#!/bin/bash

./disentangledLB_resnet_local.sh 1 0 > log23.txt 2>&1 &
./disentangledLB_resnet_local.sh 2 0 > log24.txt 2>&1 &
./disentangledLB_resnet_local.sh 3 1 > log25.txt 2>&1 &
./disentangledLB_resnet_local.sh 4 1 > log26.txt 2>&1 &
