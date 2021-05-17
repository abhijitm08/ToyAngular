#!/bin/python

import sys, os
import numpy as np
np.random.seed(2)

#############
fitdir    = os.path.abspath(os.getcwd())+'/../'
partition = 'express'
numnodes  = '1'
numtasks  = '1'
numcpus   = '1'
njobs     = 100
ncluster  = 5

#have fixed a0gplus (highly correlated)
floated_ff_nominal = ['a0f0','a0fplus','a0fperp','a0g0','a1f0','a1fplus','a1fperp','a1g0','a1gplus','a1gperp']
floated_ff_CT      = ['a0hplus','a0hperp','a0htildeplus','a1hplus','a1hperp','a1htildeplus','a1htildeperp']
#wcmin = -0.05
#wcmax =  0.05

#wc properties
wcpropbs = {}
wcpropbs['CVRScan'] = (floated_ff_nominal, -0.025, 0.02, njobs)  #started from -1 to 1
wcpropbs['CSRScan']  = (floated_ff_nominal, -0.12, 0.12, njobs)
wcpropbs['CSLScan'] = (floated_ff_nominal, -0.2, 0.2, njobs)
wcpropbs['CTScan']  = (floated_ff_nominal+floated_ff_CT, -0.05, 0.05, njobs)

#seed for the toy generation
seed = 1

for suffix in list(wcpropbs.keys()):
    ##########
    if not os.path.exists(fitdir+'plots/'+suffix+'/'):
        print('Making suffix dir', fitdir+'plots/'+suffix+'/')
        os.system('mkdir -p '+fitdir+'plots/'+suffix+'/')

    if not os.path.exists(fitdir+'plots/'+suffix+'/log/'):
        print('Making log dir', fitdir+'plots/'+suffix+'/log/')
        os.system('mkdir -p '+fitdir+'plots/'+suffix+'/log/')

    ff_float_str = ' '.join(wcpropbs[suffix][0])
    wcvals       = np.linspace(wcpropbs[suffix][1], wcpropbs[suffix][2], wcpropbs[suffix][3])
    wcname       = suffix.replace('Scan','')

    command = ''
    for indx in range(0,njobs+1):
        command  += "python LbToLclnu_fit_binscheme.py -f None -nf 20 -sf "+suffix+"_"+str(indx)+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -s "+str(seed)+" -e "+ff_float_str
        if indx != len(wcvals):
            command  += " -fpv \"{'"+wcname+"':"+str(wcvals[indx])+"}\""

        command  += "\n"
        #print(command)
        if ((indx%ncluster == 0) and (indx != 0) and indx != njobs) or (indx == njobs-1):
            uniquename= suffix+'_'+str(indx)
            logname   = fitdir+'plots/'+suffix+'/log/log_'+uniquename+'.log'
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
                #file1.write('#SBATCH --exclude=farm-wn[01-03],farm-wn[07-08],farm-wn[09-16],farm-wn[21-25],farm-wn[27-30],farm-wn41,farm-wn[71-78],farm-wn[91-92]\n')
                #file1.write('#SBATCH --nodelist=farm-wn82')
                #file1.write('#SBATCH --nodelist=farm-wn[04-06],farm-wn[81-82]')
                #file1.write('#SBATCH --mem-per-cpu=10G\n')
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
    #############
