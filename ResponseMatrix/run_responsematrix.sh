#!/bin/bash

for i in CVR CSR CSL CT SM 
do 
	for j in `seq 0 99` 
	do 
		echo sbatch -o log/response-$i-$j.log -p express --exclude=farm-wn[91-92] --mem-per-cpu=1G /disk/users/amathad/miniconda3/bin/conda run -n py3tf2 python responseMatrix_np.py $i $j
		sbatch -o log/response-$i-$j.log -p express --exclude=farm-wn[91-92] --mem-per-cpu=1G /disk/users/amathad/miniconda3/bin/conda run -n py3tf2 python responseMatrix_np.py $i $j
	done
done
