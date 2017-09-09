#!/bin/bash

cd ../tools
matlab <<< "name='disentangled_final_A10_TV0_lr0.0002_none_id1'; evaluation" >> log71.txt &
matlab <<< "name='disentangled_final_A10_TV0_lr0.0002_none_id2'; evaluation" >> log72.txt &
matlab <<< "name='disentangled_final_A10_TV0_lr0.0002_none_id3'; evaluation" >> log73.txt &

matlab <<< "name='disentangled_final_A20_TV0_lr0.0002_none_id1'; evaluation" >> log74.txt &
matlab <<< "name='disentangled_final_A20_TV0_lr0.0002_none_id2'; evaluation" >> log75.txt &
matlab <<< "name='disentangled_final_A20_TV0_lr0.0002_none_id3'; evaluation" >> log76.txt &

matlab <<< "name='disentangled_final_A80_TV0_lr0.0002_none_id1'; evaluation" >> log81.txt &
matlab <<< "name='disentangled_final_A80_TV0_lr0.0002_none_id2'; evaluation" >> log82.txt &

