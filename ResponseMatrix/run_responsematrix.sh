#!/bin/bash

#for i in CVR CSR CSL CT SM 
#do 
#	for j in `seq 0 99` 
#	do 
#		echo sbatch -o log/response-$i-$j.log -p express --exclude=farm-wn[91-92] --mem-per-cpu=1G /disk/users/amathad/miniconda3/bin/conda run -n py3tf2 python responseMatrix_np.py $i $j one_sigma
#		sbatch -o log/response-$i-$j.log -p express --exclude=farm-wn[91-92] --mem-per-cpu=1G /disk/users/amathad/miniconda3/bin/conda run -n py3tf2 python responseMatrix_np.py $i $j one_sigma
#		#echo sbatch -o log/response-$i-$j-conservative.log -p express --exclude=farm-wn[91-92] --mem-per-cpu=1G /disk/users/amathad/miniconda3/bin/conda run -n py3tf2 python responseMatrix_np.py $i $j large
#		#sbatch -o log/response-$i-$j-conservative.log -p express --exclude=farm-wn[91-92] --mem-per-cpu=1G /disk/users/amathad/miniconda3/bin/conda run -n py3tf2 python responseMatrix_np.py $i $j large
#	done
#done

indx=0
for i in CVR CSR CSL CT SM 
do 
	for j in `seq 0 99` 
	do 
		if !(($indx%20))
		then
    			echo $indx
			wait
		fi
		#echo "python responseMatrix_np.py $i $j one_sigma > log/response-$i-$j.log 2>&1 & python responseMatrix_np.py $i $j large > log/response-$i-$j-conservative.log 2>&1 &"
		#python responseMatrix_np.py $i $j one_sigma > log/response-$i-$j.log 2>&1 & python responseMatrix_np.py $i $j large > log/response-$i-$j-conservative.log 2>&1 &
		echo "python responseMatrix_np.py $i $j one_sigma > log/response-$i-$j.log 2>&1 &"
		python responseMatrix_np.py $i $j one_sigma > log/response-$i-$j.log 2>&1 &
		let "indx+=1"
	done
done
