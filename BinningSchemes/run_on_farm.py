#!/bin/python

import sys, os
sys.path.append(os.path.abspath(os.getcwd()))
from Binning_Scheme import defing_binning_scheme
curdir    = os.path.abspath(os.getcwd()) 

partition = 'express'
numnodes  = '1'
numtasks  = '1'
numcpus   = '1'

schemes  = [] 
#schemes += ['Scheme0'] 
#schemes += ['Scheme1'] 
#schemes += ['Scheme2'] 
#schemes += ['Scheme3'] 
#schemes += ['Scheme4'] 
schemes += ['Scheme5'] 
#schemes += ['Scheme6'] 

for scheme in schemes:
    BinScheme  = defing_binning_scheme()
    total_bins = (len(BinScheme[scheme]['qsq']) - 1) * (len(BinScheme[scheme]['cthl']) - 1)
    for binnum in range(total_bins):
        uniquename = scheme+'_'+str(binnum)
        logname    = curdir+'/log/log_'+uniquename+'.log'
        #command    = 'python Cache_Integrals.py -s '+scheme+' -b '+str(binnum)
        command    = 'python Cache_Integrals_ho.py -s '+scheme+' -b '+str(binnum)

        # Writing submission file
        with open('slurm_scripts/'+uniquename+'.sh', 'w') as file1:
            file1.write('#!/bin/bash\n')
            file1.write('#SBATCH --job-name='+uniquename+'\n')
            file1.write('#SBATCH --output='+logname+'\n')
            file1.write('#SBATCH --partition='+partition+'\n')
            file1.write('#SBATCH --nodes='+numnodes+'\n')
            file1.write('#SBATCH --ntasks='+numtasks+'\n')
            file1.write('#SBATCH --cpus-per-task='+numcpus+'\n')
            file1.write('#SBATCH --mem-per-cpu=10G\n')
            file1.write('\n')
            file1.write('cd '+curdir+'\n')
            file1.write('source ~/farm.sh\n')
            file1.write('conda activate farmpy3tf2\n')
            file1.write('\n')
            file1.write(command+'\n')
            file1.write('\n')
            file1.write('conda deactivate\n')
            file1.write('conda deactivate\n')
