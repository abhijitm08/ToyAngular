#!/bin/python

import sys, os
sys.path.append(os.path.abspath(os.getcwd()))
from run_fit_farm import suffixes
fitdir    = os.path.abspath(os.getcwd())+'/../'

for suffix in suffixes:
    if 'CVROnly' in suffix:
        command  = "python plot_toy_result.py -f CVR -b Scheme2 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e None -smin 0 -smax 500"
    elif 'FFOnly' in suffix:
        command  = "python plot_toy_result.py -f None -b Scheme2 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e All -smin 0 -smax 500"
    elif 'CVRFF' in suffix:
        command  = "python plot_toy_result.py -f CVR -b Scheme2 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e All -smin 0 -smax 500"
    elif 'CVRsubsetFF' in suffix:
        command  = "python plot_toy_result.py -f CVR -b Scheme2 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a1f0 a1g0 a1gperp -smin 0 -smax 500"
    elif 'FFsubset' in suffix:
        command  = "python plot_toy_result.py -f None -b Scheme2 -sf "+suffix+" -d "+fitdir+"plots/"+suffix+"/ -e a0f0 a1f0 a1fplus a1fperp a1g0 -smin 0 -smax 500"

    if 'largeyld' in suffix:
        command += " -n 75000000"
    elif 'largeryld' in suffix:
        command += " -n 750000000"

    print(command)
    os.system(command)
