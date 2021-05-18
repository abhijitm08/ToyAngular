#!/bin/python

import sys, os
import numpy as np
np.random.seed(2)

#############
fitdir    = os.path.abspath(os.getcwd())+'/'
partition = 'express'
numnodes  = '1'
numtasks  = '1'
numcpus   = '1'
njobs     = 500
ncluster  = 50

wcnames  = ['CVR', 'CSR', 'CSL', 'CT']
bschemes = ['Scheme'+str(i) for i in range(7)]
print(bschemes)

command   = ''
for wcname in wcnames:
    for bscheme in bschemes:
        for seed in list(range(njobs)):
            command  += "python LbToLclnu_fit_binscheme.py -f "+wcname+" -b "+bscheme+" -nf 20 -d ./plots/bschemes/ -p False -s "+str(seed)+" -e None"
            command  += "\n"
            print(command)
            if ((seed%ncluster == 0) and (seed != 0) and seed != njobs) or (seed == njobs-1):
                uniquename = wcname+'_'+bscheme+'_'+str(seed)
                logname    = fitdir+'plots/bschemes/log/log_'+uniquename+'.log'
                # Writing submission file
                with open('slurm_scripts/'+uniquename+'.sh', 'w') as file1:
                    file1.write('#!/bin/bash\n')
                    file1.write('#SBATCH --job-name='+uniquename+'\n')
                    file1.write('#SBATCH --output='+logname+'\n')
                    file1.write('#SBATCH --partition='+partition+'\n')
                    file1.write('#SBATCH --nodes='+numnodes+'\n')
                    file1.write('#SBATCH --ntasks='+numtasks+'\n')
                    file1.write('#SBATCH --cpus-per-task='+numcpus+'\n')
                    file1.write('#SBATCH --exclude=farm-wn[91-92]\n')
                    #file1.write('#SBATCH --mem-per-cpu=7G\n')
                    file1.write('\n')
                    file1.write('cd '+fitdir+'\n')
                    file1.write('source ~/farm.sh\n')
                    file1.write('conda activate farmpy3tf2\n')
                    file1.write('\n')
                    file1.write(command+'\n')
                    file1.write('\n')
                    file1.write('conda deactivate\n')
                    file1.write('conda deactivate\n')
            
                #print(command)
                command = ''
