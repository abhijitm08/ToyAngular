#!/bin/python

import os

fitdir    = os.path.abspath(os.getcwd())+'/'
partition = 'express'
numnodes  = '1'
numtasks  = '1'
numcpus   = '1'
ncluster  = 10

scenarios = ['CVR', 'CT', 'CSL', 'CSR', 'SM']
seed 	  = 100
nfits	  = 20
plotdir   = fitdir+'plot_modeldependency/plots/'
fitresdir = fitdir+'plot_modeldependency/fitresults/'

def make_command_nominal(scenario):
    command  = 'python LbToLclnu_modeldependency.py'
    command += ' -s '+str(seed)
    command += ' -nf '+str(nfits)
    command += ' -d '+fitresdir
    command += ' -pd '+plotdir
    command += ' -p False'
    command += ' -utgen False'
    command += ' -resn True' #nominal response matrix map is used
    command += ' -effn True' #nominal efficiency map is used
    if scenario == 'SM':
        command += ' -f None'
        command += ' -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp'
    else:
        command += ' -f '+scenario
        command += ' -e None'
    
    suffix = scenario+'_nomeffTrue_nomresTrue'
    command += ' -sf '+suffix
    return command

def make_command(scenario, model_indx, nominal_eff, nominal_responsematrix):
    command  = 'python LbToLclnu_modeldependency.py'
    command += ' -s '+str(seed)
    command += ' -nf '+str(nfits)
    command += ' -d '+fitresdir
    command += ' -pd '+plotdir
    command += ' -p False'
    command += ' -utgen False'
    if nominal_eff and not nominal_responsematrix:
        command += ' -effn True' #nominal efficiency map is used
        command += ' -resn True -resp '+fitdir+'../ResponseMatrix/responsematrix_pickled/responsematrix_'+scenario+'_'+str(model_indx)+'.p'
        suffix   = scenario+'_'+str(model_indx)+'_nomeffTrue_nomresFalse'
    elif not nominal_eff and nominal_responsematrix:
        command += ' -effn True -effp '+fitdir+'../Eff/eff_pickled/Eff_Tot_'+scenario+'_'+str(model_indx)+'.p'
        command += ' -resn True' #nominal response matrix map is used
        suffix   = scenario+'_'+str(model_indx)+'_nomeffFalse_nomresTrue'
    elif not nominal_eff and not nominal_responsematrix:
        command += ' -effn True -effp '+fitdir+'../Eff/eff_pickled/Eff_Tot_'+scenario+'_'+str(model_indx)+'.p'
        command += ' -resn True -resp '+fitdir+'../ResponseMatrix/responsematrix_pickled/responsematrix_'+scenario+'_'+str(model_indx)+'.p'
        suffix   = scenario+'_'+str(model_indx)+'_nomeffFalse_nomresFalse'
    elif nominal_eff and nominal_responsematrix:
        command += ' -resn True' #nominal response matrix map is used
        command += ' -effn True' #nominal efficiency map is used
        suffix   = 'nominal_nomeffTrue_nomresTrue'
    
    if scenario == 'SM':
        command += ' -f None'
        command += ' -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp'
    else:
        command += ' -f '+scenario
        command += ' -e None'
    
    command += ' -sf '+suffix
    return command

def write_slurmscripts(commands, uniquename_suffix):
    for i, command in enumerate(commands):
        uniquename = uniquename_suffix+'_'+str(i)
        logname    = fitdir+'log/log_'+uniquename+'.log'
        with open('slurm_scripts/'+uniquename+'.sh', 'w') as file1:
            file1.write('#!/bin/bash\n')
            file1.write('#SBATCH --job-name='+uniquename+'\n')
            file1.write('#SBATCH --output='+logname+'\n')
            file1.write('#SBATCH --partition='+partition+'\n')
            file1.write('#SBATCH --nodes='+numnodes+'\n')
            file1.write('#SBATCH --ntasks='+numtasks+'\n')
            file1.write('#SBATCH --cpus-per-task='+numcpus+'\n')
            file1.write('#SBATCH --exclude=farm-wn[91-92]\n')
            file1.write('#SBATCH --mem-per-cpu=1G\n')
            file1.write('\n')
            file1.write('cd '+fitdir+'\n')
            file1.write('source ~/farm.sh\n')
            file1.write('conda activate py3tf2\n')
            file1.write('\n')
            file1.write(command+'\n')
            file1.write('\n')
            file1.write('conda deactivate\n')
            file1.write('conda deactivate\n')

def nonNominal_model_run(nominal_eff, nominal_responsematrix, nmodels = 100):
    #nomeff = False and nomresponse = True
    commands = []
    command  = ''
    indx     = 0
    for scenario in scenarios:
        for model_indx in range(nmodels):
            cmnd  = make_command(scenario, model_indx, nominal_eff = nominal_eff, nominal_responsematrix = nominal_responsematrix)
            if ((indx+1)%ncluster == 0):
                command += cmnd+'\n'
                commands+= [command]
                command  = ''
            else:
                command += cmnd+'\n'

            indx += 1

    return commands

def main():
    #nominal: nomeff = True and nomresponse = True
    commands = []
    command  = ''
    for scenario in scenarios:
        cmnd      = make_command_nominal(scenario)
        command  += cmnd+'\n'
    commands   = [command]
    uniquename = 'nominal'
    write_slurmscripts(commands, uniquename)

    uniquename = 'model_nomeffFalse_nomresTrue'
    commands    = nonNominal_model_run(nominal_eff = False, nominal_responsematrix = True, nmodels = 100)
    write_slurmscripts(commands, uniquename)

    uniquename = 'model_nomeffTrue_nomresFalse'
    commands    = nonNominal_model_run(nominal_eff = True, nominal_responsematrix = False, nmodels = 100)
    write_slurmscripts(commands, uniquename)

    uniquename = 'model_nomeffFalse_nomresFalse'
    commands    = nonNominal_model_run(nominal_eff = False, nominal_responsematrix = False, nmodels = 100)
    write_slurmscripts(commands, uniquename)

if __name__ == '__main__':
    main()
