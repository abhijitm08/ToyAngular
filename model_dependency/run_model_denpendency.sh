#!/bin/bash

for i in Signal_MU Signal_MD Gen 
do 
	for j in CVR CSR CSL CT SM 
	do 
		echo sbatch -o log/model_dependency-$i-$j.log -p express --exclude=farm-wn[91-92] --mem-per-cpu=12G /disk/users/amathad/miniconda3/bin/conda run -n py3tf2 python model_dependency_weights.py $i $j
	done
done
