#!/bin/python

import sys, os

#############
fitdir    = os.path.abspath(os.getcwd())+'/../'
partition = 'express'
numnodes  = '1'
numtasks  = '1'
numcpus   = '1'

njobs    = 500
ncluster = 10

suffixes   = []
#suffixes  += ['CVROnly_gradFalse']
#suffixes  += ['CVROnly_gradTrue']
#suffixes  += ['CVROnly_gradFalse_largeyld']
#suffixes  += ['CVROnly_gradTrue_largeyld']
#suffixes  += ['CVROnly_gradFalse_largeryld']
#suffixes  += ['CVROnly_gradTrue_largeryld']

#suffixes  += ['FFOnly_gradFalse']
#suffixes  += ['FFOnly_gradTrue']
suffixes  += ['FFOnly_gradFalse_largeyld']
suffixes  += ['FFOnly_gradFalse_largeryld']
suffixes  += ['FFOnly_gradTrue_largeyld']
suffixes  += ['FFOnly_gradTrue_largeryld']

#suffixes  += ['CVRFF_gradFalse']
#suffixes  += ['CVRFF_gradTrue']

#suffixes  += ['CVRFF_gradFalse_notRandomised'] #to see if large forced pos. def go away (as seen before)
#suffixes  += ['CVRFF_gradTrue_largeyld']       #to see if some biases go away (75M instead of 7.5M)
#suffixes  += ['CVRFF_gradTrue_largeryld']      #to see if some biases go away (750M instead of 7.5M)
#suffixes  += ['CVRsubsetFF_gradFalse']         #to see if subset of FF params are well behaved when grad is False

for suffix in suffixes:
    #####
    print(fitdir+'plots/'+suffix+'/log/')
    if not os.path.exists(fitdir+'plots/'+suffix+'/log/'):
        print('Making log dir', fitdir+'plots/'+suffix+'/log/')
        os.system('mkdir -p '+fitdir+'plots/'+suffix+'/log/')

    command = ''
    for seed in range(0,njobs+1):
        if 'CVROnly' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CVR -b Scheme2 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e None"
        elif 'FFOnly' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f None -b Scheme2 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e All"
        elif 'CVRFF' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CVR -b Scheme2 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e All"
        elif 'CVRsubsetFF' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CVR -b Scheme2 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a1f0 a1g0 a1gperp"

        if 'gradFalse' in suffix:
            command += " -g False"

        if 'largeyld' in suffix:
            command += " -n 75000000"
        elif 'largeryld' in suffix:
            command += " -n 750000000"

        if 'notRandomised' in suffix:
            command += " -rFF False"

        command  += " -s "+str(seed)
        command  += "\n"
        #print(command)
        if (seed%ncluster == 0):
            if seed != 0:
                uniquename= suffix+'_'+str(seed)
                logname   = fitdir+'/plots/'+suffix+'/log/log_'+uniquename+'.log'
                # Writing submission file
                with open('slurm_scripts/'+uniquename+'.sh', 'w') as file1:
                    file1.write('#!/bin/bash\n')
                    file1.write('#SBATCH --job-name='+uniquename+'\n')
                    file1.write('#SBATCH --output='+logname+'\n')
                    file1.write('#SBATCH --partition='+partition+'\n')
                    file1.write('#SBATCH --nodes='+numnodes+'\n')
                    file1.write('#SBATCH --ntasks='+numtasks+'\n')
                    file1.write('#SBATCH --cpus-per-task='+numcpus+'\n')
                    file1.write('#SBATCH --exclude=farm-wn[01-03],farm-wn[07-08],farm-wn[09-16],farm-wn[21-25],farm-wn[27-30],farm-wn41,farm-wn[71-78],farm-wn[91-92]\n')
                    #file1.write('#SBATCH --exclude=farm-wn91,farm-wn92\n')
                    #file1.write('#SBATCH --nodelist=farm-wn82')
                    #file1.write('#SBATCH --nodelist=farm-wn[04-06],farm-wn[81-82]')
                    #file1.write('#SBATCH --mem-per-cpu=10G\n')
                    file1.write('\n')
                    file1.write('cd '+fitdir+'\n')
                    file1.write('source ~/farm.sh\n')
                    file1.write('conda activate py3tf2\n')
                    file1.write('\n')
                    file1.write(command+'\n')
                    file1.write('\n')
                    file1.write('conda deactivate\n')
                    file1.write('conda deactivate\n')
    
                command = ''
    #############
