import sys, os
import numpy as np
import matplotlib.pyplot as plt
###############################

#read results from files
n = 1
uncert = []
val = []
for i in range(n):
    with  open('plots/results_CVR_' + str(i) + '_Scheme6_7500000_None_toy.txt','r') as txt:
        data = txt.read()
    print(data)
    data = data.split()
    val.append(float(data[1]))
    uncert.append(float(data[2]))

print('Values for fitted CVR:')
print(val)
print('Uncertainties for fitted CVR:')
print(uncert)

print('Finished composing the results!')

#plot the results

fig, ax = plt.subplots()
ax = plt.hist(val)
ax = plt.title('Distribution of values of CVR for 100 Fits')
ax = plt.xlabel('CVR value')

fig1, ax = plt.subplots()
ax = plt.hist(uncert)
ax = plt.title('Distribution of uncertainties of CVR for 100 Fits')
ax = plt.xlabel('Hess err')

plt.show()

