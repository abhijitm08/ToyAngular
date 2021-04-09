#!/bin/python

import sys, os

##############
#curdir    = os.path.abspath(os.getcwd()) 
#partition = 'express'
#numnodes  = '1'
#numtasks  = '1'
#numcpus   = '1'
#nodelist  = ['farm-wn03','farm-wn04','farm-wn05','farm-wn06','farm-wn81','farm-wn82']
#
#njobs    = 500
#ncluster = 10
#
##suffix   = 'sigbkg_5x5bscheme_ffall'
##suffix   = 'sigbkg_5x5bscheme_ffall_largeyld'
##suffix   = 'sigbkg_5x5bscheme_ffsubset'
#
#command = ''
#for seed in range(0,njobs+1):
#    if suffix == 'sigbkg_5x5bscheme_ffall':
#        command  += "python LbToLclnu_fit_binscheme.py -f None -b Scheme2 -nf 50 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e All"
#    elif suffix == 'sigbkg_5x5bscheme_ffall_largeyld':
#        #increase yld by factor 10
#        command  += "python LbToLclnu_fit_binscheme.py -f None -b Scheme2 -n 75000000 -nf 50 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e All"
#    elif suffix == 'sigbkg_5x5bscheme_ffsubset':
#        #float only non correlated params (a0f0 a1f0 a1fperp a1g0 a1gperp a1fplus)
#        command  += "python LbToLclnu_fit_binscheme.py -f None -b Scheme2 -nf 50 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a1f0 a1fperp a1g0 a1gperp a1fplus"
#
#    command  += " -s "+str(seed)
#    command  += "\n"
#
#    if (seed%ncluster == 0):
#        if seed != 0:
#            uniquename= suffix+'_'+str(seed)
#            logname   = curdir+'/plots/'+suffix+'/log/log_'+uniquename+'.log'
#            # Writing submission file
#            with open('slurm_scripts/'+uniquename+'.sh', 'w') as file1:
#                file1.write('#!/bin/bash\n')
#                file1.write('#SBATCH --job-name='+uniquename+'\n')
#                file1.write('#SBATCH --output='+logname+'\n')
#                file1.write('#SBATCH --partition='+partition+'\n')
#                file1.write('#SBATCH --nodes='+numnodes+'\n')
#                file1.write('#SBATCH --ntasks='+numtasks+'\n')
#                file1.write('#SBATCH --cpus-per-task='+numcpus+'\n')
#                file1.write('#SBATCH --exclude=farm-wn91,farm-wn92\n')
#                file1.write('#SBATCH --nodelist=farm-wn82')
#                #file1.write('#SBATCH --nodelist=farm-wn[04-06],farm-wn[81-82]')
#                #file1.write('#SBATCH --mem-per-cpu=10G\n')
#                file1.write('\n')
#                file1.write('cd '+curdir+'\n')
#                file1.write('source ~/farm.sh\n')
#                file1.write('conda activate py3tf2\n')
#                file1.write('\n')
#                file1.write(command+'\n')
#                file1.write('\n')
#                file1.write('conda deactivate\n')
#                file1.write('conda deactivate\n')
#
#            command = ''
##############


suffix   = 'sigbkg_5x5bscheme_ffall'
#suffix   = 'sigbkg_5x5bscheme_ffall_largeyld'
#suffix   = 'sigbkg_5x5bscheme_ffsubset'

if suffix == 'sigbkg_5x5bscheme_ffall':
    command  = "python plot_toy_result.py -f None -b Scheme2 -sf "+suffix+" -d ./plots/"+suffix+"/ -e All -smin 0 -smax 500"
elif suffix == 'sigbkg_5x5bscheme_ffall_largeyld':
    command  = "python plot_toy_result.py -f None -b Scheme2 -n 75000000 -sf "+suffix+" -d ./plots/"+suffix+"/ -e All -smin 0 -smax 500"
elif suffix == 'sigbkg_5x5bscheme_ffsubset':
    command  = "python plot_toy_result.py -f None -b Scheme2 -sf "+suffix+" -d ./plots/"+suffix+"/ -e a0f0 a1f0 a1fperp a1g0 a1gperp a1fplus -smin 0 -smax 500"

print(command)
