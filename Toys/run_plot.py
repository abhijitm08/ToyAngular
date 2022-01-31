#!/bin/python

import sys, os
sys.path.append(os.path.abspath(os.getcwd()))
from run_fit_farm import suffixes
fitdir    = os.path.abspath(os.getcwd())+'/../'

for suffix in suffixes:
    if 'CVROnly' in suffix:
        command  = "python plot_toy_result.py -f CVR -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e None -smin 0 -smax 500"
    elif 'CSL_effFalse_resFalse' in suffix:
        command  = "python plot_toy_result.py -f CSL -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e None -smin 0 -smax 500"
    elif 'CSL_effTrue_resTrue' in suffix:
        command  = "python plot_toy_result.py -f CSL -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e None -smin 0 -smax 500"
    elif 'CSL_subsetFF1' in suffix:
        command  = "python plot_toy_result.py -f CSL -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a1gperp a0g0 a1f0 a1g0 a1fplus a1gplus a0fplus -smin 0 -smax 500"
    elif 'CSL_subsetFF2' in suffix:
        command  = "python plot_toy_result.py -f CSL -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a1fplus a1gplus a0fplus -smin 0 -smax 500"
    elif 'CSL_Scheme2_subsetFF1' in suffix:
        command  = "python plot_toy_result.py -f CSL -b Scheme2 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a1gperp a0g0 a1f0 a1g0 a1fplus a1gplus a0fplus -smin 0 -smax 500"
    elif 'CSL_Scheme2_subsetFF2' in suffix:
        command  = "python plot_toy_result.py -f CSL -b Scheme2 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a1fplus a1gplus a0fplus -smin 0 -smax 500"
    elif 'CSL_Scheme6_subsetFF1' in suffix:
        command  = "python plot_toy_result.py -f CSL -b Scheme6 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a1gperp a0g0 a1f0 a1g0 a1fplus a1gplus a0fplus -smin 0 -smax 500"
    elif 'CSL_Scheme6_subsetFF2' in suffix:
        command  = "python plot_toy_result.py -f CSL -b Scheme6 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a1fplus a1gplus a0fplus -smin 0 -smax 500"
    elif 'CSL_NP1_subsetFF1' in suffix:
        command  = "python plot_toy_result.py -f CSL -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a1gperp a0g0 a1f0 a1g0 a1fplus a1gplus a0fplus -smin 0 -smax 500"
    elif 'CSL_NP1_subsetFF2' in suffix:
        command  = "python plot_toy_result.py -f CSL -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a1fplus a1gplus a0fplus -smin 0 -smax 500"
    elif 'CSL_NP1_FF' in suffix:
        command  = "python plot_toy_result.py -f CSL -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -smin 0 -smax 500"
    elif 'CSR_NP1_FF' in suffix:
        command  = "python plot_toy_result.py -f CSR -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -smin 0 -smax 500"
    elif 'CVR_NP1_FF' in suffix:
        command  = "python plot_toy_result.py -f CVR -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -smin 0 -smax 500"
    elif 'CT_NP1_FF' in suffix:
        command  = "python plot_toy_result.py -f CT -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -smin 0 -smax 500"
    elif 'CSL_NP2_subsetFF1' in suffix:
        command  = "python plot_toy_result.py -f CSL -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a1gperp a0g0 a1f0 a1g0 a1fplus a1gplus a0fplus -smin 0 -smax 500"
    elif 'CSL_NP2_subsetFF2' in suffix:
        command  = "python plot_toy_result.py -f CSL -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a1fplus a1gplus a0fplus -smin 0 -smax 500"
    elif 'CSL_NP2_FF' in suffix:
        command  = "python plot_toy_result.py -f CSL -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -smin 0 -smax 500"
    elif 'CSR_NP2_FF' in suffix:
        command  = "python plot_toy_result.py -f CSR -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -smin 0 -smax 500"
    elif 'CVR_NP2_FF' in suffix:
        command  = "python plot_toy_result.py -f CVR -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -smin 0 -smax 500"
    elif 'CT_NP2_FF' in suffix:
        command  = "python plot_toy_result.py -f CT -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -smin 0 -smax 500"
    elif 'FFOnly' in suffix:
        command  = "python plot_toy_result.py -f None -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -smin 0 -smax 500"
    elif 'CVRFF' in suffix:
        command  = "python plot_toy_result.py -f CVR -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -smin 0 -smax 500"
    elif 'CSRFF' in suffix:
        command  = "python plot_toy_result.py -f CSR -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -smin 0 -smax 500"
    elif 'CSLFF' in suffix:
        command  = "python plot_toy_result.py -f CSL -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp -smin 0 -smax 500"
    elif 'CTFF' in suffix:
        command  = "python plot_toy_result.py -f CT -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a0fplus a0fperp a0g0 a1f0 a1fplus a1fperp a1g0 a1gplus a1gperp a0hplus a0hperp a0htildeplus a1hplus a1hperp a1htildeplus a1htildeperp -smin 0 -smax 500"
    elif 'CVRsubsetFF' in suffix:
        command  = "python plot_toy_result.py -f CVR -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a1f0 a1g0 a1gperp -smin 0 -smax 500"
    elif 'FFsubset' in suffix:
        command  = "python plot_toy_result.py -f None -b Scheme5 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a1f0 a1fplus a1fperp a1g0 -smin 0 -smax 500"

    if 'largeyld' in suffix:
        command += " -n 75000000"
    elif 'largeryld' in suffix:
        command += " -n 750000000"

    print(command)
    os.system(command)
