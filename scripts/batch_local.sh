#!/bin/bash

./disentangled_final_local.sh 1 5 0 > log68.txt 2>&1
./disentangled_final_local.sh 2 5 0 > log69.txt 2>&1
./disentangled_final_local.sh 3 5 0 > log70.txt 2>&1

./disentangled_final_local.sh 1 40 0 > log77.txt 2>&1
./disentangled_final_local.sh 2 40 0 > log78.txt 2>&1
./disentangled_final_local.sh 3 40 0 > log79.txt 2>&1

./disentangled_final_local.sh 1 80 0 > log80.txt 2>&1

bash evaluate_all1.sh
