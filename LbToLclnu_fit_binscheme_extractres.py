import sys, os
import numpy as np
import scipy as sc
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
###############################

def g(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))

#read results from files
n = 100
coeff = 'CVR'
scheme = 6
FF = 'All'
TCVR = 0
uncert = []
val = []
for i in range(n):
    with  open('plots/results_' + coeff + '_' + str(i+100) + '_Scheme' + str(scheme) + '_7500000_' + FF + '_toy.txt','r') as txt:
        data = txt.read()
    print(data)
    data = data.split()
    ii = data.index(coeff)
    val.append(float(data[ii+1]))
    uncert.append(float(data[ii+2]))
    print(ii, data[ii+1], data[ii+2])

    #i1 = data.index('has_parameters_at_limit')
    #i2 = data.index('has_posdef_covar')
    

print('Values for fitted CVR:')
print(val)
print('Uncertainties for fitted CVR:')
print(uncert)

print('Finished composing the results!')

#plot the results
fig, ax = plt.subplots(2)
ax[0].hist(val)
fig.suptitle('Distribution of values and Hessian errors of CVR for '+ str(n) +' Fits in Binning Scheme '+ str(scheme))
ax[0].set_xlabel('CVR value')
ax[0].set_ylabel('Number of Fits [#]')

ax[1].hist(uncert)
ax[1].set_xlabel('Hessian Error')
ax[1].set_ylabel('Number of Fits [#]')
#ax[1].set_xticks([5.55e-4,5.56e-4,5.57e-4])
#ax[1].set_xticklabels(['5.55e-4','5.56e-4','5.57e-4'])
#plt.show()


#fig, ax = plt.subplots()
#gauss = np.random.normal(loc = TCVR, scale = 1, size = np.size(np.array(val)))
#newval = (np.array(val)-TCVR)/np.array(uncert)
#ax.hist(gauss, alpha = 0.5, label = 'Normal (0,1)')
#countnew, binsnew, ignored = ax.hist(newval, alpha = 0.5, label = 'Values')
#ax.legend()
#ax.set_xlabel('Hessian Error')
#ax.set_ylabel('Bincount/Error')
#ax.set_title('(CVR - TrueCVR)/Hess')
#plt.show()

plt.figure()
val = np.array(val)
newval = (np.array(val)-TCVR)/np.array(uncert)
count, bins, ignored = plt.hist(newval, 30, density=True, label = 'Data', alpha = 0.8)
x = np.linspace(np.min(newval), np.max(newval), len(count))
print(x, count, bins)
#popt, pcov = curve_fit(g, x, count, p0 = [np.mean(count),np.std(count)])
popt, pcov = curve_fit(g, x, count, p0 = [TCVR,1])
print('Fitted Mean and Sigma:',popt)
plt.plot(x, g(x,*popt), label = 'Fitted Normal', color = 'r')
plt.plot(x, g(x,TCVR,1), label = 'Normal N(0,1)')
plt.legend()
plt.xlabel('Hessian Error')
plt.ylabel(coeff + ' - True' + coeff + ')/Hess [%/100]')
plt.title('(' + coeff + ' - True' + coeff + ')/Hess of binscheme '+ str(scheme) + ' and ' + FF + ' FF floated' )
plt.show()

#evaluate mean and median
meanval = np.mean(val)
meanunc = np.mean(uncert)
medianval = np.median(val)
medianuncert = np.median(uncert)

print('Mean: ', meanval, meanunc)
print('Median: ', medianval, medianuncert)
