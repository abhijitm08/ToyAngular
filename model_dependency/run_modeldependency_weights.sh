#!/bin/bash

#for i in Signal_MU Signal_MD Gen 
#do 
#	for j in CVR CSR CSL CT SM 
#	do 
#		echo sbatch -o log/model_dependency_weights/model_dependency-$i-$j.log -p express --exclude=farm-wn[91-92] --mem-per-cpu=12G /disk/users/amathad/miniconda3/bin/conda run -n py3tf2 python model_dependency_weights.py $i $j one_sigma
#		#sbatch -o log/model_dependency_weights/model_dependency-$i-$j.log -p express --exclude=farm-wn[91-92] --mem-per-cpu=12G /disk/users/amathad/miniconda3/bin/conda run -n py3tf2 python model_dependency_weights.py $i $j one_sigma
#		echo sbatch -o log/model_dependency_weights/model_dependency-$i-$j-conservative.log -p express --exclude=farm-wn[91-92] --mem-per-cpu=12G /disk/users/amathad/miniconda3/bin/conda run -n py3tf2 python model_dependency_weights.py $i $j large
#		#sbatch -o log/model_dependency_weights/model_dependency-$i-$j-conservative.log -p express --exclude=farm-wn[91-92] --mem-per-cpu=12G /disk/users/amathad/miniconda3/bin/conda run -n py3tf2 python model_dependency_weights.py $i $j large
#	done
#done


#1st batch
{ python model_dependency_weights.py Signal_MU CVR one_sigma > log/model_dependency_weights/model_dependency-Signal_MU-CVR.log  2>&1; python model_dependency_weights.py Signal_MU CVR large > log/model_dependency_weights/model_dependency-Signal_MU-CVR-conservative.log  2>&1; }   &
{ python model_dependency_weights.py Signal_MU CSR one_sigma > log/model_dependency_weights/model_dependency-Signal_MU-CSR.log  2>&1; python model_dependency_weights.py Signal_MU CSR large > log/model_dependency_weights/model_dependency-Signal_MU-CSR-conservative.log  2>&1; }  &
{ python model_dependency_weights.py Signal_MU CSL one_sigma > log/model_dependency_weights/model_dependency-Signal_MU-CSL.log  2>&1; python model_dependency_weights.py Signal_MU CSL large > log/model_dependency_weights/model_dependency-Signal_MU-CSL-conservative.log  2>&1; }  &
{ python model_dependency_weights.py Signal_MU CT one_sigma  > log/model_dependency_weights/model_dependency-Signal_MU-CT.log   2>&1; python model_dependency_weights.py Signal_MU CT large  > log/model_dependency_weights/model_dependency-Signal_MU-CT-conservative.log  2>&1; }  &
{ python model_dependency_weights.py Signal_MU SM one_sigma  > log/model_dependency_weights/model_dependency-Signal_MU-SM.log   2>&1; python model_dependency_weights.py Signal_MU SM large  > log/model_dependency_weights/model_dependency-Signal_MU-SM-conservative.log  2>&1; }  &
wait
#2nd batch
{ python model_dependency_weights.py Signal_MD CVR one_sigma > log/model_dependency_weights/model_dependency-Signal_MD-CVR.log  2>&1; python model_dependency_weights.py Signal_MD CVR large > log/model_dependency_weights/model_dependency-Signal_MD-CVR-conservative.log  2>&1; }  &
{ python model_dependency_weights.py Signal_MD CSR one_sigma > log/model_dependency_weights/model_dependency-Signal_MD-CSR.log  2>&1; python model_dependency_weights.py Signal_MD CSR large > log/model_dependency_weights/model_dependency-Signal_MD-CSR-conservative.log  2>&1; }  &
{ python model_dependency_weights.py Signal_MD CSL one_sigma > log/model_dependency_weights/model_dependency-Signal_MD-CSL.log  2>&1; python model_dependency_weights.py Signal_MD CSL large > log/model_dependency_weights/model_dependency-Signal_MD-CSL-conservative.log  2>&1; }  &
{ python model_dependency_weights.py Signal_MD CT one_sigma  > log/model_dependency_weights/model_dependency-Signal_MD-CT.log   2>&1; python model_dependency_weights.py Signal_MD CT large  > log/model_dependency_weights/model_dependency-Signal_MD-CT-conservative.log  2>&1; }  &
{ python model_dependency_weights.py Signal_MD SM one_sigma  > log/model_dependency_weights/model_dependency-Signal_MD-SM.log   2>&1; python model_dependency_weights.py Signal_MD SM large  > log/model_dependency_weights/model_dependency-Signal_MD-SM-conservative.log  2>&1; }  &
wait
#3rd batch
{ python model_dependency_weights.py Gen CVR one_sigma       > log/model_dependency_weights/model_dependency-Gen-CVR.log        2>&1; python model_dependency_weights.py Gen CVR large       > log/model_dependency_weights/model_dependency-Gen-CVR-conservative.log  2>&1; }  &
{ python model_dependency_weights.py Gen CSR one_sigma       > log/model_dependency_weights/model_dependency-Gen-CSR.log        2>&1; python model_dependency_weights.py Gen CSR large       > log/model_dependency_weights/model_dependency-Gen-CSR-conservative.log  2>&1; }  &
{ python model_dependency_weights.py Gen CSL one_sigma       > log/model_dependency_weights/model_dependency-Gen-CSL.log        2>&1; python model_dependency_weights.py Gen CSL large       > log/model_dependency_weights/model_dependency-Gen-CSL-conservative.log  2>&1; }  &
{ python model_dependency_weights.py Gen CT one_sigma        > log/model_dependency_weights/model_dependency-Gen-CT.log         2>&1; python model_dependency_weights.py Gen CT large        > log/model_dependency_weights/model_dependency-Gen-CT-conservative.log  2>&1; }  &
{ python model_dependency_weights.py Gen SM one_sigma        > log/model_dependency_weights/model_dependency-Gen-SM.log         2>&1; python model_dependency_weights.py Gen SM large        > log/model_dependency_weights/model_dependency-Gen-SM-conservative.log  2>&1; }  &
