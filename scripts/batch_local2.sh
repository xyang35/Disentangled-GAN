#!/bin/bash

./disentangled_final_local.sh 1 10 1 > log71.txt 2>&1
./disentangled_final_local.sh 2 10 1 > log72.txt 2>&1
./disentangled_final_local.sh 3 10 1 > log73.txt 2>&1

./disentangled_final_local.sh 1 20 1 > log74.txt 2>&1
./disentangled_final_local.sh 2 20 1 > log75.txt 2>&1
./disentangled_final_local.sh 3 20 1 > log76.txt 2>&1

./disentangled_final_local.sh 2 80 1 > log81.txt 2>&1
./disentangled_final_local.sh 3 80 1 > log82.txt 2>&1

bash evaluate_all2.sh
