import sys, os
import numpy as np
import scipy as sc
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
###############################
#Usage: python LbToLclnu_fit_binscheme_extractres.py <'coeff(CVR,CT,CSR,CSL)'> <scheme_min(0,1,2,3,4,5,6)> <scheme_max(0,1,2,3,4,5,6),included> <'FormFactorsFLoated FF(None, All)'>
###############################


def g(x, mu, sigma): #simple gaussian
    print('Sigma:', sigma)
    print('Mean:', mu)
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))

def extractdat(j, m1, m2, coeff, FF, n):
    val_tmp = []
    uncert_tmp = []

    for i in range(n):
        #with  open('plots/results_' + coeff + '_' + str(i+100) + '_Scheme' + str(j) + '_7500000_' + FF + '_toy.txt','r') as txt:
        with open('/disk/lhcb_data/amathad/forHelena/ToyAngular/plots/bschemes/results_' + coeff + '_' + str(i) + '_Scheme' + str(j) + '_7500000_' + FF + '_toy.txt','r') as txt:
            data = txt.read()
        data = data.split()
        ii = data.index(coeff)
        val_tmp.append(float(data[ii+1]))
        uncert_tmp.append(float(data[ii+2]))
        #print(data[ii+1], data[ii+2])

    return val_tmp, uncert_tmp

def gaussfit(newval, TWC, stdguess):
    count, bins, ignored = plt.hist(newval, 40, density=True, label = 'Data', alpha = 0.8)
    x = np.linspace(np.min(newval), np.max(newval), len(count))
    popt, pcov = curve_fit(g, x, count, p0 = [TWC,stdguess])
    print('Fitted Mean and Sigma:',popt)

    return count, bins, popt, pcov, x 


def main():
    #give inputs to function
    n = 500 
    TWC = 0
    stdguess1 = 1
    check = 0

    #read given inputs
    coeff = sys.argv[1]
    m1 = int(sys.argv[2])
    m2 = int(sys.argv[3])+1
    FF = sys.argv[4]

     #number of interations thru binschemes, e.g. m = 6: binschemes 0,1,2,3,4,5 
    d = m2-m1
    binscheme = range(m1,m2)
    print('Mode: ', np.array(binscheme), d) #either loop or single mode

    #adapt the guess of the standard deviation for the fit
    stdguess = np.ones(m2-m1)*stdguess1
    print('Stdguess: ', stdguess)

    #read results from files
    val = []
    uncert = []
    val_tmp = []
    uncert_tmp = []
    val_tot = np.ndarray(shape = (d,n))
    uncert_tot = np.ndarray(shape = (d,n))
    opt_std = []
    opt_mean = []
    opt_cov = []

    for j in range(d):
        print('Working on Binscheme: ', binscheme[j])
        val_tmp, uncert_tmp = extractdat(binscheme[j], m1, m2, coeff, FF, n)
        print('Valtmp:', np.size(val_tmp))
        print('Uncerttmp:' , np.size(uncert_tmp))
        if d == 1:
            val = np.array(val_tmp)
            uncert = np.array(uncert_tmp)
        else:
            val_tot[j][:] = np.array(val_tmp)
            uncert_tot[j][:] = np.array(uncert_tmp)
            #take only one line from val, uncert
            val = np.array(val_tot[j][:])
            uncert = np.array(uncert_tot[j][:])
        
        print('Finished composing the results!')

        #plot the results
        if 1 == 0:
            fig, ax = plt.subplots(2)
            ax[0].hist(val)
            fig.suptitle('Distribution of values and Hessian errors of CVR for '+ str(n) +' Fits in Binning Scheme '+ str(binscheme[j]))
            ax[0].set_xlabel('CVR value')
            ax[0].set_ylabel('Number of Fits [#]')

            ax[1].hist(uncert)
            ax[1].set_xlabel('Hessian Error')
            ax[1].set_ylabel('Number of Fits [#]')
            #ax[1].set_xticks([5.55e-4,5.56e-4,5.57e-4])
            #ax[1].set_xticklabels(['5.55e-4','5.56e-4','5.57e-4'])
            plt.show()

            fig, ax = plt.subplots()
            gauss = np.random.normal(loc = TWC, scale = 1, size = np.size(val))
            newval = (val-TWC)/uncert
            ax.hist(gauss, alpha = 0.5, label = 'Normal (0,1)')
            countnew, binsnew, ignored = ax.hist(newval, alpha = 0.5, label = 'Values')
            ax.legend()
            ax.set_xlabel('Hessian Error')
            ax.set_ylabel('Bincount/Error')
            ax.set_title('(CVR - TrueCVR)/Hess')
            #plt.show()

        plt.figure()
        newval = (np.array(val)-TWC)/np.array(uncert)
        #newval = val-TWC
        print('Newval: ',newval)
        if d == 1:
            count, bins, popt, pcov, x = gaussfit(newval, TWC, stdguess1)
        else:
            count, bins, popt, pcov, x = gaussfit(newval, TWC, stdguess[j])

        #count, bins, ignored = plt.hist(newval, 40, density=True, label = 'Data', alpha = 0.8)
        #x = np.linspace(np.min(newval), np.max(newval), len(count))
        #popt, pcov = curve_fit(g, x, count, p0 = [TWC,stdguess[j]])

        opt_mean.append(popt[0])
        opt_std.append(popt[1])
        opt_cov.append(pcov)

        if d == 1:
            opt_mean = opt_mean[0]
            opt_std = opt_std[0]
            opt_cov = opt_cov[0]

        #print('Fitted Mean and Sigma:',popt)
        plt.plot(x, g(x, *popt), label = 'Fitted Normal N('+ str(np.round(popt[0],5)) + ','+ str(np.round(popt[1],5)) + ')', color = 'r')
        plt.plot(x, g(x, TWC, stdguess[j]), label = 'Normal N(0,' + str(stdguess[j]) + ')')
        plt.legend()
        plt.xlabel('Hessian Error')
        plt.ylabel(coeff + ' - True' + coeff + ')/Hess [%/100]')
        #plt.ylabel(coeff + ' - True' + coeff)
        plt.title('(' + coeff + ' - True' + coeff + ')/Hess of binscheme '+ str(binscheme[j]) + ' and ' + FF + ' FF floated' )
        #plt.title(coeff + ' in binscheme '+ str(binscheme[j]) + ' and ' + FF + ' FF floated') 

        print('plots/results_' + coeff + '_' + str(100) + '_Scheme' + str(binscheme[j]) + '_7500000_' + FF + '_toy.txt')

    plt.show()
    exit(1)
    #write composed data to .txt file
    #f = open('data_combined/Results_' + coeff + '_binschemes.txt' , 'w')   #'w' for writing
    #f.write(str(val_tot.tolist()) + str( uncert_tot.tolist()))    # Write inside file
    #f.close()
    
    #Plot 0: Define layout of plot and combine plots
    if 1 == 1:
        if d == 1:
            print('No second plot: d = 1')
        else:
            fig, ax = plt.subplots(d, sharex = True, sharey = True)
            for d2 in range(d):
                newval = np.array(val_tot[d2][:])-TWC
                fig.suptitle(coeff + '- True' + coeff + ' in Binning Scheme ' + str(m1) + ' - ' + str(m2-1) + ' with ' + FF + ' FF floated')
                ax[d2].hist(newval, 40, density = True)
                ax[d2].plot(x,g(x, opt_mean[d2], opt_std[d2]), label = 'N(' + str(np.round(opt_mean[d2],5)) + ',' + str(np.round(opt_std[d2],5)) + ')', color = 'r')
                ax[d2].set_ylabel('Scheme ' + str(binscheme[d2]))
                ax[d2].legend()
            ax[d2].set_xlabel(coeff)
            plt.show()
    
    if 1 == 0:
        fig, ax = plt.subplots(2, sharex = True)
        ax[0].plot(binscheme, opt_std, 'x', linewidth = 3)
        ax[1].plot(binscheme, opt_mean, 'x', linewidth = 3)
        ax[1].hlines(0, m1-1, m2, linewidth = 2)
        ax[0].grid()
        ax[1].grid()
        #ax[1].set_xticks(bins)
        #ax[1].set_xticklabels(bins)
        ax[1].set_xlabel('Binning Scheme')
        for xy in zip(binscheme, np.round(opt_std,5)):
            ax[0].annotate('(%s, %s)' % xy, xy = xy)
        for xy in zip(binscheme, np.round(opt_mean,5)):
            ax[1].annotate('(%s, %s)' % xy, xy = xy)
        ax[0].set_ylabel('Standard deviation')
        ax[1].set_ylabel('Mean')
        ax[1].set_xlim([binscheme[0], binscheme[-1]])
        fig.suptitle('Standard deviation and mean of '+ coeff +' distribution with '+ FF +' FF floated')
        plt.show()

    #Plot 1: How many Sigmas away from TWC?
    if 1 == 1:
        fig,ax = plt.subplots()
        err = opt_std/np.sqrt(n)
        sigmas = opt_mean/err
        print('Values Plot1 (binscheme,mean,std,err):',binscheme,opt_mean,opt_std,err,'Sigmas:', sigmas)
        ax.errorbar(binscheme, opt_mean, yerr = err, fmt = 'x')
        ax.grid()
        ax.hlines(0,m1-1,m2, linestyle = 'dashed', color = 'k')
        for a in range(d):
        #    ax.annotate( '(' + str(np.round(opt_mean[a],6)) + '±' + str(np.round(err[a],6)) + ')', xy = (a,opt_mean[a]))
            if d == 1:
                print('Plot1:','(' + str(np.round(sigmas,2))+ ' σ)',binscheme[a],opt_mean)
                ax.annotate( '(' + str(np.round(sigmas,2))+ ' σ)', xy = (binscheme[a],opt_mean))
            else:    
                ax.annotate( '(' + str(np.round(sigmas[a],2))+ ' σ)', xy = (binscheme[a],opt_mean[a]))
        ax.set_ylabel(coeff + ' - True' + coeff)
        ax.set_xlabel('Binning Scheme')
        ax.set_title('Mean and Standard Error of '+ coeff +' distribution with '+ FF +' FF floated')
        plt.show()

    #Plot 2: Plot StdDev seperately
    if 1 == 1:
        fig, ax = plt.subplots()
        std_err = opt_std/np.sqrt(2*n - 2)
        ax.errorbar(binscheme, opt_std, yerr = std_err, fmt = 'x')
        for a in range(d):
            if d == 1:
                print('Plot2:','(' + str(np.round(opt_std,5))+ ' σ)',binscheme[a],opt_std)
                ax.annotate( '(' + str(np.round(opt_std,5))+ ')', xy = (binscheme[a],opt_std))
            else:
                ax.annotate( '(' + str(np.round(opt_std[a],5))+ ')', xy = (binscheme[a],opt_std[a]))
        #ax.hlines(0,m1-1,m2, linestyle = 'dashed', color = 'k')
        ax.set_ylabel('StdDev')
        ax.set_xlabel('Binning Scheme')
        ax.set_title('Standard Deviation of '+ coeff +' with '+ FF +' FF floated')
        ax.grid()
        plt.show()

    #Plot 3: Degradation Plot 
    if binscheme[-1] == 6 and d != 1:
        fig, ax = plt.subplots()
        deg = (1-(opt_std/opt_std[-1]))*100
        print('Degradation:',deg)
        #print('Shape deg, binscheme[-1]:',np.shape(deg), binscheme[-1])
        ax.plot(binscheme, deg, 'x')
        #ax.legend()
        ax.grid()
        ax.hlines(0, m1-1, m2, color = 'k', linestyle = 'dashed')
        for a in range(d):
            print('range:',a, deg[a])
            print('xy = ', binscheme[a], deg[a])
            print('('+str(np.round(deg[a],3))+' %)')
            ax.annotate( '(' + str(np.round(deg[a],3)) + ' %)', xy = (binscheme[a],deg[a]))
        ax.set_xlabel('Binning Schemes')
        ax.set_ylabel('[1 - StdDev(Scheme(i))/StdDev(Scheme6)] * 100 [%]')
        ax.set_title('Standard Deviation for different WCs \wrt Binning Scheme 6 for '+FF+' floated FF')
        plt.show()

    #Check Degradation Plot for binschemes 2 and 6 for 500 Fits: Use with code for Scheme 2 (single)
    if check == 1:
        print('Degradation Check starting...')
        bin_check = 6
        val_check, uncert_check = extractdat(bin_check, bin_check, bin_check+1, coeff, FF, n)
        val_check = np.array(val_check)
        uncert_check = np.array(val_check)
        newval_check = val_check - TWC
        count_c, bins_c, popt_c, pcov_c, x = gaussfit(newval_check, TWC, stdguess1)
        opt_mean_c = popt_c[0]
        opt_std_c = popt_c[1]
        deg = (1-(opt_std/opt_std_c))*100
        
        fig, ax = plt.subplots()
        ax.plot(binscheme, deg, 'x')
        #ax.legend()
        ax.grid()
        ax.hlines(0, m1-1, m2, color = 'k', linestyle = 'dashed')
        for a in range(d):
            print('range:',a, deg)
            print('xy = ', binscheme[a], deg)
            print('('+str(np.round(deg,3))+' %)')
            ax.annotate( '(' + str(np.round(deg,3)) + ' %)', xy = (binscheme[a],deg))
        ax.set_xlabel('Binning Schemes')
        ax.set_ylabel('[1 - StdDev(Scheme(2))/StdDev(Scheme6)] * 100 [%]')
        ax.set_title('Standard Deviation for CVR Binning Scheme 2 \wrt Binning Scheme 6 for '+FF+' floated FF')
        plt.show()

    if 1 == 0:
        fig, ax = plt.subplots()
        ax.boxplot(val_tot.T)
        ax.hlines(0,m1+1,m2)
        ax.set_xticklabels(binscheme)
        ax.set_ylabel('Pull Mean')
        ax.set_title('Mean and Standard deviation of '+ coeff +' in binschemes '+ str(m1) + ' - ' + str(m2) + ' with ' + FF + ' FF floated')
        ax.set_xlabel('Binning Scheme')
        ax.grid()
        plt.show()

    exit(1)

    #########
   # Extract different WC
    #########
 
    coeffs = ['CVR', 'CT', 'CSR', 'CSL']
    val_wcs = np.zeros((len(coeffs), d, n))
    uncert_wcs = np.zeros((len(coeffs), d, n))

    opt_mean_wcs = np.zeros((len(coeffs),d))
    opt_std_wcs = np.zeros((len(coeffs),d))

    for i in range(len(coeffs)):
        for j in range(m1,m2):
            val_tmp2, uncert_tmp2 = extractdat(j,m1,m2,coeffs[i],FF,n)
            val_wcs[i][j][:] = np.array(val_tmp2)
            uncert_wcs[i][j][:] = np.array(uncert_tmp2)
            
            newval2 = np.array(val_wcs[i][j][:]) - TWC
            count2, bins2, popt_wcs, pcov_wcs, x = gaussfit(newval2, TWC, stdguess[j])
            opt_mean_wcs[i][j] = popt_wcs[0]
            opt_std_wcs[i][j] = popt_wcs[1]

    #if 1 == 1:
        #val_wcs_std = np.std(val_wcs, 2)
        #val_wcs_mean = np.mean(val_wcs,2)
        #uncert_wcs_std = np.std(uncert_wcs, 2)
        #uncert_wcs_mean = np.mean(uncert_wcs,2)
        #plt.figure()
        

    if 1 == 0:
        plt.figure()
        for i in range(len(coeffs)):
            err_wcs = opt_std_wcs[i][:].flatten()/np.sqrt(n)
            plt.errorbar(binscheme, opt_mean_wcs[i][:].flatten(), yerr = err_wcs, fmt = 'x', linewidth = (6-i)/2, label = coeffs[i])
        plt.legend()
        plt.hlines(0, m1, m2, color = 'k', linestyle = 'dashed')
        plt.grid()
        plt.xlabel('Binning Schemes')
        plt.ylabel('WC - TrueWC')
        plt.title('Mean and Error on the Mean for different WCs and Binning Schemes for '+FF+' FF floated')
        plt.show()

    #Plot 3: Combined Degradation Plot
    if 1 == 1:
        fig, ax = plt.subplots()
        deg = np.zeros((len(coeffs),d))
        for i in range(len(coeffs)):
            deg[i][:] = (1-(opt_std_wcs[i][:]/opt_std_wcs[i][-1]))*100
            #print('Shape deg, binscheme[-1]:',np.shape(deg), binscheme[-1])
            ax.plot(binscheme, deg[i][:].flatten(), 'x', label = coeffs[i])
            for a in range(d):
                print('range:',a, deg[i][a])
                print('xy = ', binscheme[a], deg[i][a])
                print('('+str(np.round(deg[i][a],0))+' %)')
                ax.annotate( '(' + str(np.round(deg[i][a],1)) + ' %)', xy = (binscheme[a],deg[i][a]))
        ax.legend()
        ax.grid()
        ax.hlines(0, m1, m2, color = 'k', linestyle = 'dashed')
        ax.set_xlabel('Binning Schemes')
        ax.set_ylabel('[1 - StdDev(Scheme(i))/StdDev(Scheme6)] * 100 [%]')
        ax.set_title('Standard Deviation for different WCs \wrt Binning Scheme 6 for '+FF+' floated FF')
        plt.show()



if __name__ == '__main__':
    main()
