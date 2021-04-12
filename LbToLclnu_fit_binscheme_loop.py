import sys, os

n = 100
for i in range(n):
    command = 'python LbToLclnu_fit_binscheme.py -f CT -b Scheme6 -inter 10 -p False -e None -s '+ str(i+100)
    os.system(command)

print('Finished with '+ str(n) +' fits!') 
