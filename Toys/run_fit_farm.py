#!/bin/python

import sys, os

#############
fitdir    = os.path.abspath(os.getcwd())+'/../'
partition = 'standard'
numnodes  = '1'
numtasks  = '1'
numcpus   = '1'

njobs    = 500
ncluster = 25

suffixes   = []
#suffixes  += ['CSL_effFalse_resFalse_gradTrue']
#suffixes  += ['CSL_effTrue_resTrue_gradTrue']
#suffixes  += ['CSL_subsetFF1_gradTrue']
#suffixes  += ['CSL_subsetFF2_gradTrue']
#suffixes  += ['CSL_NP1_subsetFF1_gradTrue'] #CSL = +1, 7 FF
#suffixes  += ['CSL_NP1_subsetFF2_gradTrue'] #CSL = +1, 3 FF
#suffixes  += ['CSL_NP1_FF_gradTrue']        #CSL = +1, 10 FF
#suffixes  += ['CSL_NP2_subsetFF1_gradTrue'] #CSL = -1, 7 FF
#suffixes  += ['CSL_NP2_subsetFF2_gradTrue'] #CSL = -1, 3 FF
#suffixes  += ['CSL_NP2_FF_gradTrue']        #CSL = +1, 10 FF

#suffixes  += ['CSR_NP1_FF_gradTrue'] #CSR = +1, 10 FF
#suffixes  += ['CVR_NP1_FF_gradTrue'] #CVR = +1, 10 FF
#suffixes  += ['CT_NP1_FF_gradTrue']  #CT  = +1, 10 FF
#suffixes  += ['CSR_NP2_FF_gradTrue'] #CSR = +1, 10 FF
#suffixes  += ['CVR_NP2_FF_gradTrue'] #CVR = +1, 10 FF
#suffixes  += ['CT_NP2_FF_gradTrue']  #CT  = +1, 10 FF
suffixes  += ['CSL_Scheme2_subsetFF1_gradTrue']
suffixes  += ['CSL_Scheme2_subsetFF2_gradTrue']
suffixes  += ['CSL_Scheme6_subsetFF1_gradTrue']
suffixes  += ['CSL_Scheme6_subsetFF2_gradTrue']

#suffixes  += ['CVROnly_gradFalse']
#suffixes  += ['CVROnly_gradTrue']
#suffixes  += ['CVROnly_gradFalse_largeyld']
#suffixes  += ['CVROnly_gradTrue_largeyld']
#suffixes  += ['CVROnly_gradFalse_largeryld']
#suffixes  += ['CVROnly_gradTrue_largeryld']

#suffixes  += ['FFOnly_gradTrue']
#suffixes  += ['FFOnly_gradFalse']
#suffixes  += ['FFOnly_gradFalse_largeyld']
#suffixes  += ['FFOnly_gradFalse_largeryld']
#suffixes  += ['FFOnly_gradTrue_largeyld']
#suffixes  += ['FFOnly_gradTrue_largeryld']
#suffixes  += ['FFsubset_gradTrue']
#suffixes  += ['FFsubset_gradFalse']

#suffixes  += ['CVRFF_gradTrue']
#suffixes  += ['CSRFF_gradTrue']
#suffixes  += ['CSLFF_gradTrue']
#suffixes  += ['CTFF_gradTrue']

#suffixes  += ['CVRFF_gradFalse_notRandomised'] #to see if large forced pos. def go away (as seen before)
#suffixes  += ['CVRFF_gradTrue_largeyld']       #to see if some biases go away (75M instead of 7.5M)
#suffixes  += ['CVRFF_gradTrue_largeryld']      #to see if some biases go away (750M instead of 7.5M)
#suffixes  += ['CVRsubsetFF_gradFalse']         #to see if subset of FF params are well behaved when grad is False

for suffix in suffixes:
    #####
    print(fitdir+'plots/'+suffix+'/log/')
    if not os.path.exists(fitdir+'plots/'+suffix):
        print('Making suffix dir', fitdir+'plots/'+suffix)
        os.system('mkdir -p '+fitdir+'plots/'+suffix)

    if not os.path.exists(fitdir+'plots/'+suffix+'/log/'):
        print('Making log dir', fitdir+'plots/'+suffix+'/log/')
        os.system('mkdir -p '+fitdir+'plots/'+suffix+'/log/')

    command = ''
    for seed in range(0,njobs+1):
        if 'CVROnly' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CVR -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e None"
        elif 'CSL_effFalse_resFalse' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSL -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn False -resn False -utgen False -e None"
        elif 'CSL_effTrue_resTrue' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSL -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e None"
        elif 'CSL_subsetFF1' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSL -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a1gperp a0g0 a1f0 a1g0 a1fplus a1gplus a0fplus"
        elif 'CSL_subsetFF2' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSL -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a1fplus a1gplus a0fplus"
        elif 'CSL_Scheme2_subsetFF1' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSL -b Scheme2 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a1gperp a0g0 a1f0 a1g0 a1fplus a1gplus a0fplus\
                         -effp /home/hep/amathad/Packages/ToyAngular/Eff/Eff_Tot_SM_Scheme2.p\
                         -resp /home/hep/amathad/Packages/ToyAngular/ResponseMatrix/responsematrix_Scheme2.p"
        elif 'CSL_Scheme2_subsetFF2' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSL -b Scheme2 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a1fplus a1gplus a0fplus\
                         -effp /home/hep/amathad/Packages/ToyAngular/Eff/Eff_Tot_SM_Scheme2.p\
                         -resp /home/hep/amathad/Packages/ToyAngular/ResponseMatrix/responsematrix_Scheme2.p"
        elif 'CSL_Scheme6_subsetFF1' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSL -b Scheme6 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a1gperp a0g0 a1f0 a1g0 a1fplus a1gplus a0fplus\
                         -effp /home/hep/amathad/Packages/ToyAngular/Eff/Eff_Tot_SM_Scheme6.p\
                         -resp /home/hep/amathad/Packages/ToyAngular/ResponseMatrix/responsematrix_Scheme6.p"
        elif 'CSL_Scheme6_subsetFF2' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSL -b Scheme6 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a1fplus a1gplus a0fplus\
                         -effp /home/hep/amathad/Packages/ToyAngular/Eff/Eff_Tot_SM_Scheme6.p\
                         -resp /home/hep/amathad/Packages/ToyAngular/ResponseMatrix/responsematrix_Scheme6.p"
        elif 'CSL_NP1_subsetFF1' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSL -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a1gperp a0g0 a1f0 a1g0 a1fplus a1gplus a0fplus -gpv \"{'CSL': 1.0}\""
        elif 'CSL_NP1_subsetFF2' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSL -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a1fplus a1gplus a0fplus -gpv \"{'CSL': 1.0}\""
        elif 'CSL_NP1_FF' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSL -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -gpv \"{'CSL': 1.0}\""
        elif 'CSR_NP1_FF' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSR -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -gpv \"{'CSR': 1.0}\""
        elif 'CVR_NP1_FF' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CVR -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -gpv \"{'CVR': 1.0}\""
        elif 'CT_NP1_FF' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CT -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -gpv \"{'CT': 1.0}\""
        elif 'CSL_NP2_subsetFF1' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSL -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a1gperp a0g0 a1f0 a1g0 a1fplus a1gplus a0fplus -gpv \"{'CSL': -1.0}\""
        elif 'CSL_NP2_subsetFF2' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSL -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a1fplus a1gplus a0fplus -gpv \"{'CSL': -1.0}\""
        elif 'CSL_NP2_FF' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSL -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -gpv \"{'CSL': -1.0}\""
        elif 'CSR_NP2_FF' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSR -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -gpv \"{'CSR': -1.0}\""
        elif 'CVR_NP2_FF' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CVR -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -gpv \"{'CVR': -1.0}\""
        elif 'CT_NP2_FF' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CT -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -gpv \"{'CT': -1.0}\""
        elif 'FFOnly' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f None -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp"
        elif 'CVRFF' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CVR -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp"
        elif 'CSRFF' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSR -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp"
        elif 'CSLFF' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CSL -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp"
        elif 'CTFF' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CT -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp a0hplus a0hperp a0htildeplus a1hplus a1hperp a1htildeplus a1htildeperp"
        elif 'CVRsubsetFF' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f CVR -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a1f0 a1g0 a1gperp"
        elif 'FFsubset' in suffix:
            command  += "python LbToLclnu_fit_binscheme.py -f None -b Scheme5 -nf 20 -sf "+suffix+" -d ./plots/"+suffix+"/ -p False -effn True -resn True -utgen False -e a0f0 a1f0 a1fplus a1fperp a1g0"

        if 'gradFalse' in suffix:
            command += " -g False"

        if 'largeyld' in suffix:
            command += " -n 75000000"
        elif 'largeryld' in suffix:
            command += " -n 750000000"

        if 'notRandomised' in suffix:
            command += " -rFF False"

        command  += " -s "+str(seed)
        command  += " -cov False"
        command  += "\n"
        #print(command)
        if (seed%ncluster == 0):
            if seed != 0:
                uniquename= suffix+'_'+str(seed)
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
                    #file1.write('#SBATCH --exclude=farm-wn91,farm-wn92\n')
                    #file1.write('#SBATCH --nodelist=farm-wn82')
                    #file1.write('#SBATCH --nodelist=farm-wn[04-06],farm-wn[81-82]')
                    #file1.write('#SBATCH --mem-per-cpu=10G\n')
                    file1.write('#SBATCH --mem-per-cpu=1G\n')
                    file1.write('\n')
                    file1.write('cd '+fitdir+'\n')
                    file1.write('source ~/farm.sh\n')
                    file1.write('conda activate farmpy3tf2\n')
                    file1.write('\n')
                    file1.write(command+'\n')
                    file1.write('\n')
                    file1.write('conda deactivate\n')
                    file1.write('conda deactivate\n')
    
                command = ''
    #############
