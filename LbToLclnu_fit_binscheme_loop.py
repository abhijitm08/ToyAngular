import sys, os

n = 100
for i in range(n):
    command = 'python LbToLclnu_fit_binscheme.py -f CVR -b Scheme6 -s '+ str(i+100)
    os.system(command)
    #cm1 = 'rm results_CVR_'+str(i)+'_Scheme6_7500000_None_toy_data.pdf'
    #cm2 = 'rm results_CVR_'+str(i)+'_Scheme6_7500000_None_toy_fit.pdf'
    #cm3 = 'rm results_CVR_'+str(i)+'_Scheme6_7500000_None_toy_pull.pdf'
    #os.system(cm1)
    #os.system(cm2)
    #os.system(cm3)

print('Finished with '+ str(n) +' fits!') 
