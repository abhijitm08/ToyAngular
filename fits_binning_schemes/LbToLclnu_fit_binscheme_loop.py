import sys, os

n = 400
for i in range(n):
    command = 'python ../LbToLclnu_fit_binscheme.py -f CVR -b Scheme6 -inter 10 -p False -e None -nf 20 -s '+ str(i+200)
    os.system(command)

print('Finished with '+ str(n) +' fits!') 
