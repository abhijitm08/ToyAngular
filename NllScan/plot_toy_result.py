from ROOT import TFile, TTree, TH1D, TH2D, gROOT, gStyle, TCanvas, kGreen, kRed, TCut, TPaveText, TLegend, kBlue, gPad, TLine, TF1, TGraph2D, TF2, TEllipse
gROOT.SetBatch(True)
gStyle.SetOptFit(1)
import numpy as np
import sys, os
import pandas as pd
import argparse
fitdir=os.path.dirname(os.path.abspath(__file__))+'/../'
import matplotlib.pyplot as plt
import seaborn as sns
from root_pandas import to_root
from scipy.interpolate import interp1d

def interpolated_intercepts(x, y1, y2):
    """Find the intercepts of two curves, given by the same x data"""

    def intercept(point1, point2, point3, point4):
        """find the intersection between two lines
        the first line is defined by the line between point1 and point2
        the first line is defined by the line between point3 and point4
        each point is an (x,y) tuple.

        So, for example, you can find the intersection between
        intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

        Returns: the intercept, in (x,y) format
        """

        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x,y

        L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
        L2 = line([point3[0],point3[1]], [point4[0],point4[1]])

        R = intersection(L1, L2)

        return R

    idxs = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)

    xcs = []
    ycs = []

    for idx in idxs:
        xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
        xcs.append(xc)
        ycs.append(yc)
    return np.array(xcs), np.array(ycs)

def main():
    ######### - just to get the headers by reading one of the toy i.e. when seed = 1
    #non floated params stored in fit results
    nonparams  = []
    nonparams += ['loglh']
    nonparams += ['is_valid']
    nonparams += ['has_parameters_at_limit']
    nonparams += ['has_accurate_covar']
    nonparams += ['has_posdef_covar']
    nonparams += ['has_made_posdef_covar']
    #floated params in fit results
    floatedvars= []

    #make fres dictionary and fill floatedvars. here just get the keys for the fres dictionary. no content stored
    fres       = {}
    resfname   = direc+'results_'+floatWC+'_'+str(seed)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'_0.txt'
    if not os.path.isfile(resfname): print('File does not exist', resfname)
    fres['indx']  = []
    fres['wcval'] = []
    f = open(resfname, "r")
    for l in f.readlines(): 
        varname = l.split()[0]
        if varname not in nonparams:
            fres[varname+'_val']  = []
            fres[varname+'_err']  = []
            fres[varname+'_pull'] = []
            floatedvars += [varname]
        else:
            fres[varname]   = []

    f.close()
    ###########

    ######### - loop through the toys and fill fres dictionary for each toy
    wcvals = np.linspace(wcval_min, wcval_max, wcnvals)
    for i in range(len(wcvals)):
        #fit result file
        resfname = direc+'results_'+floatWC+'_'+str(seed)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'_'+str(i)+'.txt'
        if not os.path.isfile(resfname): 
            print('File does not exist', resfname)
            continue

        #open the file and store the results into fres dictionary
        fres['indx']  += [i]
        fres['wcval'] += [wcvals[i]]
        f = open(resfname, "r")
        ls= f.readlines()
        for l in ls: 
            varname = l.split()[0]
            if varname not in nonparams:
                tval    = true_val[varname]
                fval    = float(l.split()[1])
                ferr    = float(l.split()[2])
                pull    = -100.
                if ferr != 0.0: pull = (fval - tval)/ferr
                fres[varname+'_val'] += [fval]
                fres[varname+'_err'] += [ferr]
                fres[varname+'_pull']+= [pull]
            else:
                if varname == 'loglh':
                    fres[varname]  += [float(l.split()[1])]
                else:
                    fres[varname]  += [int(l.split()[1])]
    
    #convert the fres dictionary into pandas data frame and make some requirements
    df_fres       = pd.DataFrame.from_dict(fres)
    shape_before  = df_fres.shape
    print('Before requirements of valid minima and params in limit', shape_before)
    condvalid     = df_fres['is_valid'] == 1
    condparmlmt   = df_fres['has_parameters_at_limit'] == 0
    condacccov    = df_fres['has_accurate_covar'] == 1
    condposdefcov = df_fres['has_posdef_covar'] == 1
    condmadeposdef= df_fres['has_made_posdef_covar'] == 0
    cond          = np.logical_and(condvalid, condparmlmt)
    cond          = np.logical_and(cond, condacccov)
    cond          = np.logical_and(cond, condposdefcov)
    cond          = np.logical_and(cond, condmadeposdef)
    df_fres       = df_fres[cond]
    shape_after   = df_fres.shape
    print('After requirements of valid minima and params in limit', shape_after)
    #if shape_before[0] != shape_after[0]:
    #    raise Exception("Shapes before and after the requirement do not agree")

    x = df_fres[cond]['wcval'].to_numpy()
    y = df_fres[cond]['loglh'].to_numpy()
    #for xi, yi in zip(x,y): print(xi,yi)
    f2   = interp1d(x, y, kind='cubic')

    xnew    = np.linspace(np.min(x), np.max(x), 10*wcnvals)
    ynew    = f2(xnew) #curve
    y_minpt_sigma = np.min(ynew) + 0.5 
    y_line  = y_minpt_sigma * np.ones_like(ynew) #line
    xcs, ycs= interpolated_intercepts(xnew,ynew,y_line)

    #idx = np.argwhere(np.diff(np.sign(ynew - y_line)) != 0)
    #plt.plot(x[idx], y1[idx], 'ms', ms=7, label='Nearest data-point method')
    y_minpt = np.min(ynew)
    x_minpt = xnew[ynew == np.min(ynew)][0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('$-\log(L)$', fontsize=10)
    ax.set_xlabel(scan_wcname, fontsize=10)
    ax.scatter(x, y, color='b')
    ax.axvline(x_minpt, color='red', linestyle='--', alpha = 0.8, label = 'Min Val'.format(x_minpt))
    ax.axvline(0., color='black', linestyle='-', alpha = 0.8, label = 'True Val')
    ax.axvspan(xcs[0][0], xcs[1][0], alpha=0.3, color='red', label = '$68\%$ CI: ${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$'.format(x_minpt, xcs[1][0] - x_minpt, x_minpt - xcs[0][0]))
    #for xc, yc in zip(xcs, ycs):
    #    print('intercept', xc, yc)
    #    ax.plot(xc, yc, 'o')
    #ax.plot(xnew, ynew, color='r')
    #ax.axhline(y_minpt_sigma, color='black', linestyle='--', alpha = 0.8)
    ax.legend(loc='best')
    print('Lower Bound: ', xcs[1][0] - x_minpt)
    print('Upper Bound: ', x_minpt - xcs[0][0])
    fig.tight_layout()
    pfname   = nlldir+'plot_scan_'+scan_wcname+'_'+floatWC+'_'+str(seed)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'_'+str(wcval_min).replace('.', 'pt')+'_'+str(wcval_max).replace('.', 'pt')+'_'+str(wcnvals)+'.pdf'
    print(pfname)
    fig.savefig(pfname, bbox_inches = 'tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for plot_toy_results.py')
    parser.add_argument('-f'      , '--floatWC'     , dest='floatWC'      ,type=str,  default='None'    , help='(string) Name of the Wilson coefficient (WC) floated when fitting. Available options are [CVR,CSR,CSL,CT,None]. Default is None.')
    parser.add_argument('-s'      , '--seed'        , dest='seed'         ,type=int,  default=int(0)    , help='(int) Value of seed (an index for the toy ) used in the generation of fake/toy data. Default is zero.')
    parser.add_argument('-b'      , '--bscheme'     , dest='bscheme'      ,type=str,  default='Scheme2' , help='(string) Binning scheme to be used when fitting. Default is Scheme2. See BinningSchemes/Binning_Scheme.py.')
    parser.add_argument('-n'      , '--nevnts'      , dest='nevnts'       ,type=int,  default=int(7.5e6), help='(int) Size of the toy sample generated. Default is 7.5M events.')
    parser.add_argument('-d'      , '--direc'       , dest='direc'        ,type=str,  default='./plots/', help='(string) Directory in which the fit result (.txt) is saved. Default is current directory.')
    parser.add_argument('-e'      , '--floated_FF'  , dest='floated_FF'   ,nargs='+', default = ['None'], help="(list) List of form factor (FF) parameters floated in fit. See LbToLclnu_fit_binscheme.py. Default is ['None'].") 
    parser.add_argument('-sf'     , '--suffix'      , dest='suffix'       ,type=str,  default='toy'     , help="(int) A unique suffix used to get the fit result file (*_suffix_*.txt). Default is 'toy'.")
    parser.add_argument('-swc'    , '--scan_wcname' , dest='scan_wcname'  ,type=str,  default='CVR'     , help='(string) Name of the Wilson coefficient (WC) that has been fixed to a certain value. Available options are [CVR,CSR,CSL,CT,None]. Default is CVR.')
    parser.add_argument('-wcmin'  , '--wcval_min'   , dest='wcval_min'    ,type=float,default=-1.       , help='(int) Min value of wc to be scanned. Default is -1.')
    parser.add_argument('-wcmax'  , '--wcval_max'   , dest='wcval_max'    ,type=float,default=-1.       , help='(int) Max value of wc to be scanned. Default is  1.')
    parser.add_argument('-wcnvals', '--wcnvals'     , dest='wcnvals'      ,type=int,  default=int(100)  , help='(int) Number of wc values used to do the scan. Default is 100.')

    args       = parser.parse_args()
    floatWC    = args.floatWC
    seed       = args.seed
    bscheme    = args.bscheme
    nevnts     = args.nevnts
    suffix     = args.suffix
    direc      = args.direc
    floated_FF = args.floated_FF
    scan_wcname= args.scan_wcname
    wcval_min  = args.wcval_min
    wcval_max  = args.wcval_max
    wcnvals    = args.wcnvals
    print(args)
    if not direc.endswith('/'): direc += '/'

    ######### set the true values used in the generation
    true_val = {}
    #WC values
    true_val["CVL"] = 0.0
    true_val["CVR"] = 0.0
    true_val["CSR"] = 0.0
    true_val["CSL"] = 0.0
    true_val[ "CT"] = 0.0
    #FF mean values
    f = open(fitdir+'/FF_cov/LambdabLambdac_results.dat', 'r')
    for l in f.readlines(): true_val[l.split()[0]] = float(l.split()[1])
    f.close()
    #true_val['a0gperp']      = true_val['a0gplus']
    #true_val['a0htildeperp'] = true_val['a0htildeplus']
    #print(true_val)
    #FF covariance matrix
    true_covariance = {}
    for n in list(true_val.keys()):
        if ('a0' in n) or ('a1' in n): true_covariance[n] = {}
    
    for l in open(fitdir+"/FF_cov/LambdabLambdac_covariance.dat", "r").readlines(): 
        true_covariance[l.split()[0]][l.split()[1]] = float(l.split()[2])
    
    #print(true_covariance)
    #########

    #make directories to store the plots
    basedir = direc+'plots/'
    nlldir  = basedir+'Nll/'
    if not os.path.exists(basedir):  os.system('mkdir '+basedir)
    if not os.path.exists(nlldir):   os.system('mkdir '+nlldir)
    main()
