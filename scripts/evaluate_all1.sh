#!/bin/bash

cd ../tools
matlab <<< "name='disentangled_final_A5_TV0_lr0.0002_none_id1'; evaluation" >> log68.txt &
matlab <<< "name='disentangled_final_A5_TV0_lr0.0002_none_id2'; evaluation" >> log69.txt &
matlab <<< "name='disentangled_final_A5_TV0_lr0.0002_none_id3'; evaluation" >> log70.txt &

matlab <<< "name='disentangled_final_A40_TV0_lr0.0002_none_id1'; evaluation" >> log77.txt &
matlab <<< "name='disentangled_final_A40_TV0_lr0.0002_none_id2'; evaluation" >> log78.txt &
matlab <<< "name='disentangled_final_A40_TV0_lr0.0002_none_id3'; evaluation" >> log79.txt &

matlab <<< "name='disentangled_final_A80_TV0_lr0.0002_none_id1'; evaluation" >> log80.txt &

