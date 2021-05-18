#!/bin/python

import sys, os
sys.path.append(os.path.abspath(os.getcwd()))
from run_fit_farm import wcpropbs, seed
fitdir  = os.path.abspath(os.getcwd())+'/../'

for suffix in list(wcpropbs.keys()):
    ff_float_str = ' '.join(wcpropbs[suffix][0])
    wc_min       = wcpropbs[suffix][1]
    wc_max       = wcpropbs[suffix][2]
    wc_nvals     = wcpropbs[suffix][3]
    wcname       = suffix.replace('Scan','')
    command      = "python plot_toy_result.py -f None -sf "+suffix+" -d ../plots/"+suffix+"/ -s "+str(seed)+" -e "+ff_float_str
    command     += " -swc "+wcname+" -wcmin "+str(wc_min)+" -wcmax "+str(wc_max)+" -wcnvals "+str(wc_nvals)
    print(command)

    os.system(command)
