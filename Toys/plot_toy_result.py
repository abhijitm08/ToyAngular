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

def str2bool(v):
    """Function used in argparse"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_ylimits(cntval, cnts, lowers, uppers, factor = 1.):
    ymin        = np.min(cnts - lowers)
    ymax        = np.max(cnts + uppers)
    dx          = ymax - ymin
    ybelow_cntr = np.abs(cntval - ymin) + 0.1 * dx * factor
    yabove_cntr = np.abs(cntval - ymax) + 0.1 * dx * factor
    ymaxabs     = max(ybelow_cntr,yabove_cntr)
    #print(cntval - ymaxabs, cntval  + ymaxabs)
    return cntval - ymaxabs, cntval  + ymaxabs

def empty_axis(ax):
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

def fill_pullaxis(ax1, floatedvars, pullmean_val, ycenter, factor = 1., color = 'r', xoffset = 0.2):
    xindx      = np.array(list(range(len(floatedvars))))
    pmcnt      = np.array([pullmean_val[k][0] for k in floatedvars])
    pmerr_lower= np.array([pullmean_val[k][1] for k in floatedvars])
    pmerr_upper= np.array([pullmean_val[k][2] for k in floatedvars])

    ax1.set_ylim(get_ylimits(ycenter, pmcnt, pmerr_lower, pmerr_upper, factor = factor))
    ax1.axhline(y=ycenter, color='black', linestyle='--', alpha = 0.8)
    for xind in xindx: ax1.axvline(x=xind, color='black', linestyle='dotted', alpha = 0.1)
    ax1.errorbar(xindx, pmcnt, yerr=[1.*pmerr_lower, 1.*pmerr_upper], fmt='s', color=color, alpha = 0.8, capsize=4, capthick=1)
    for x, y, yerr in zip(xindx, pmcnt, pmerr_lower): ax1.text(x+xoffset, y, r'{0:1.2f}$\sigma$'.format(abs((y-ycenter)/yerr)), fontsize=7, color=color, alpha=0.8)
    ax1.set_xticks(xindx, minor=False)
    ax1.set_xticklabels(floatedvars, fontdict=None, rotation=45, ha='right', minor=False)

def fill_mnstd_axis(ax1, floatedvars, pullmean_val, pullsigma_val, ycenter, plottype = 'pull', color = 'r'):
    xindx      = np.array(list(range(len(floatedvars))))
    pmcnt      = np.array([pullmean_val[k][0] for k in floatedvars])
    pscnt      = np.array([pullsigma_val[k][0] for k in floatedvars])

    ax1.set_ylim(get_ylimits(ycenter, pmcnt, pscnt, pscnt))
    ax1.axhline(y=ycenter, color='black', linestyle='--', alpha = 0.8)
    if plottype == 'pull':
        ax1.axhline(y= 1.,color='green', linestyle='dotted', alpha = 0.8)
        ax1.axhline(y=-1, color='green', linestyle='dotted', alpha = 0.8)

    for xind in xindx: ax1.axvline(x=xind, color='black', linestyle='dotted', alpha = 0.1)
    ax1.errorbar(xindx, pmcnt, yerr=[1.*pscnt, 1.*pscnt], fmt='s', color=color, alpha = 0.8, capsize=4, capthick=1)
    ax1.set_xticks(xindx, minor=False)
    ax1.set_xticklabels(floatedvars, fontdict=None, rotation=45, ha='right', minor=False)

def fill_stddevaxis(ax2, fvars2, diffsigma_val, ycenter, plottype = 'lqcd', factor = 1.):
    xindx2           = np.array(list(range(len(fvars2))))
    dfm_err_lower2    = np.array([diffsigma_val[k][0] for k in fvars2])
    if plottype == 'lqcd':
        dfm_other_lower2= np.array([diffsigma_val[k][1] for k in fvars2])
    elif plottype == 'fit':
        dfm_other_lower2= np.array([diffsigma_val[k][2] for k in fvars2])
    else:
        raise Exception('Plottype not recognised')

    #print('FFnames Toys_sigmastat LQCD_sigmastat')
    #for i in range(len(fvars2)):print('{0} {1} {2}'.format(fvars2[i], dfm_err_lower2[i], dfm_other_lower2[i]))

    merrother = dfm_err_lower2/dfm_other_lower2
    ax2.set_ylim(get_ylimits(ycenter, merrother, np.zeros_like(merrother), np.zeros_like(merrother), factor = factor))
    ax2.axhline(y=ycenter, color='black', linestyle='--', alpha = 0.8)
    for xind in xindx2: ax2.axvline(x=xind, color='black', linestyle='dotted', alpha = 0.1)
    ax2.errorbar(xindx2, merrother, fmt='o', color='b', alpha = 0.8, capsize=4, capthick=1)
    ax2.set_xticks(xindx2, minor=False)
    ax2.set_xticklabels(fvars2, fontdict=None, rotation=45, ha='right', minor=False)

def getlqcd_cov(ffvars, getcorr = False):
        true_covariance3= {}
        for k1 in ffvars:
            true_covariance3[k1]= {}
            for k2 in ffvars:
                true_covariance3[k1][k2]= true_covariance[k1][k2]

        if getcorr:
            true_corr3 = {}
            for k1 in ffvars:
                true_corr3[k1] = {}
                for k2 in ffvars:
                    true_corr3[k1][k2]= true_covariance3[k1][k2]/np.sqrt(true_covariance3[k1][k1] * true_covariance3[k2][k2]) 

            df_truecor = pd.DataFrame.from_dict(true_corr3)
            return df_truecor
        else:
            df_truecov = pd.DataFrame.from_dict(true_covariance3)
            return df_truecov

def get_covmatrices(floatedvars, df_fres_posdef, getcorr = False, getlqcd = False):
    #make toy and fit covariance dictionary
    fit_covariance3_tmp = {}
    toy_covariance3 = {}
    for k1 in floatedvars:
        fit_covariance3_tmp[k1] = {}
        toy_covariance3[k1] = {}
        for k2 in floatedvars:
            fit_covariance3_tmp[k1][k2] = []

    #get the covariance matrix file
    for i in df_fres_posdef['seed'].to_numpy():
        #get the file
        covresfname = direc+'results_'+floatWC+'_'+str(i)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'_covmatrix.txt'
        if not os.path.isfile(covresfname): 
            print('File does not exist', covresfname)
            continue

        #store the covariance matrix result
        f = open(covresfname, "r")
        ls= f.readlines()
        for l in ls: 
            par1 = l.split()[0]
            par2 = l.split()[1]
            parval = float(l.split()[2])
            if (par1 in floatedvars) and (par2 in floatedvars):
                #print(floatedvars)
                #print(par1, par2)
                fit_covariance3_tmp[par1][par2] += [float(l.split()[2])]

        f.close()

    #get the mean of the fit covariance entries and get the covariance matrix from toys
    fit_covariance3 = {}
    for k1 in floatedvars:
        fit_covariance3[k1] = {}
        for k2 in floatedvars:
            fit_covariance3[k1][k2] = np.mean(fit_covariance3_tmp[k1][k2])
            toy_covariance3[k1][k2] = ((df_fres_posdef[k1+'_val'] - df_fres_posdef[k1+'_val'].mean()) * (df_fres_posdef[k2+'_val'] - df_fres_posdef[k2+'_val'].mean())).mean()

    output_tup = None
    if getcorr:
        fit_corr3, toy_corr3 = {}, {}
        for k1 in floatedvars:
            fit_corr3[k1] = {}
            toy_corr3[k1] = {}
            for k2 in floatedvars:
                fit_corr3[k1][k2] = fit_covariance3[k1][k2]/np.sqrt(fit_covariance3[k1][k1] * fit_covariance3[k2][k2]) 
                toy_corr3[k1][k2] = toy_covariance3[k1][k2]/np.sqrt(toy_covariance3[k1][k1] * toy_covariance3[k2][k2]) 

        df_fitcor  = pd.DataFrame.from_dict(fit_corr3 )
        df_toycor  = pd.DataFrame.from_dict(toy_corr3 )
        output_tup = (df_fitcor, df_toycor)
    else:
        df_fitcov  = pd.DataFrame.from_dict(fit_covariance3)
        df_toycov  = pd.DataFrame.from_dict(toy_covariance3)
        output_tup = (df_fitcov, df_toycov)
        
    if getlqcd: 
        #only applies to form factors
        ffvars  = [k for k in floatedvars if 'a0' in k or 'a1' in k]
        output_tup += (getlqcd_cov(ffvars, getcorr = getcorr),)

    return output_tup
        
def plotcorr(ax, df, vmin = -1., vmax = 1., ticks = None, center = 0.):
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)]= True
    heatmap = sns.heatmap(df, mask = mask, square = True, linewidths = 0.5, cmap = 'coolwarm',cbar_kws = {'shrink': .4, 'ticks' : ticks}, vmin = vmin, vmax = vmax, center = center, annot = True,annot_kws = {'size': 12})
    ax.set_yticklabels(df.columns, rotation = 0)
    ax.set_xticklabels(df.columns)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

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
    resfname   = direc+'results_'+floatWC+'_0_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.txt'
    if not os.path.isfile(resfname): print('File does not exist', resfname)
    fres['seed'] = []
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
    for i in range(seedmin, seedmax+1):
        #fit result file
        resfname = direc+'results_'+floatWC+'_'+str(i)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.txt'
        if not os.path.isfile(resfname): 
            print('File does not exist', resfname)
            continue

        #open the file and store the results into fres dictionary
        fres['seed'] += [i]
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
    df_fres   = pd.DataFrame.from_dict(fres)
    print('Before requirements of valid minima and params in limit', df_fres.shape[0])
    condvalid     = df_fres['is_valid'] == 1
    condparmlmt   = df_fres['has_parameters_at_limit'] == 0
    cond          = np.logical_and(condvalid, condparmlmt)
    df_fres       = df_fres[cond]
    print('After requirements of valid minima and params in limit', df_fres.shape[0])

    #seperate the fit results into the ones that are forced pos. def. covariance matrix and ones that are not 
    condacccov    = df_fres['has_accurate_covar'] == 1
    condposdefcov = df_fres['has_posdef_covar'] == 1
    condmadeposdef= df_fres['has_made_posdef_covar'] == 0
    cond          = np.logical_and(condacccov, condposdefcov)
    cond          = np.logical_and(cond, condmadeposdef)
    df_fres_posdef    = df_fres[ cond]
    df_fres_nonposdef = df_fres[np.logical_not(cond)]
    print('After nonposdef requirements', df_fres_nonposdef.shape)
    print('After posdef requirements'   , df_fres_posdef.shape)
    #fraction of forced pos def.
    nfrac   = float(df_fres_nonposdef.shape[0])/df_fres.shape[0]

    pullmean_val  = {}
    pullsigma_val = {}
    diffmean_val  = {}
    diffsigma_val = {}
    for k in list(df_fres.keys()):
        if 'pull' in k:
            limitpull = 5.
            condlimit2= np.abs(df_fres_nonposdef[k]) < limitpull
            condlimit3= np.abs(df_fres_posdef[k]) < limitpull
            vals2     = df_fres_nonposdef[condlimit2][k].to_numpy()
            vals3     = df_fres_posdef[condlimit3][k].to_numpy()
            #print('Abs of ', k, 'required to be below', limitpull)
            #print('After this req. for posdef', vals3.shape)
            #print('After this req. for nonposdef', vals2.shape)
    
            hst = TH1D("hpull"+k, "hpull"+k, 50, -limitpull, limitpull)
            hst.SetXTitle(k)
            hst.SetYTitle("Freq")
            hst.SetLineColor(4)
            hst.SetLineStyle(1)
            hst.SetLineWidth(3)
            for val in vals3: hst.Fill(val)
            for val in vals2: hst.Fill(val)
    
            hst3 = TH1D("h3pull"+k, "h3pull"+k, 50, -limitpull, limitpull)
            hst3.SetXTitle(k)
            hst3.SetYTitle("Freq")
            hst3.SetLineColor(2)
            hst3.SetLineStyle(9)
            hst3.SetLineWidth(3)
            for val3 in vals3: hst3.Fill(val3)
            hst2 = TH1D("h2pull"+k, "h2pull"+k, 50, -limitpull, limitpull)
            hst2.SetXTitle(k)
            hst2.SetYTitle("Freq")
            hst2.SetLineColor(8)
            hst2.SetLineStyle(9)
            hst2.SetLineWidth(3)
            for val2 in vals2: hst2.Fill(val2)
    
            c1 = TCanvas("c1pull"+k,"c1pull"+k, 600, 400)
            hst.Draw("")
            hst2.Draw("same")
            hst3.Draw("same")
            #Fit only the status 3 fits
            fit = hst.Fit("gausn")
            myfunc = hst.GetFunction("gausn")
            pullmean_val[k.replace('_pull', '')]  = (myfunc.GetParameter(1), myfunc.GetParError(1), myfunc.GetParError(1))
            pullsigma_val[k.replace('_pull', '')] = (myfunc.GetParameter(2), myfunc.GetParError(2), myfunc.GetParError(1))
            pfname = pulldir+'plot_'+k+'_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
            c1.SaveAs(pfname)
        elif '_val' in k and 'is_valid' not in k:
            #print(k)
            #true_val
            tval = true_val[k.replace('_val', '')]

            #get difference b/w true 
            vals = df_fres[k].to_numpy()   - tval
            vals2= df_fres_nonposdef[k].to_numpy() - tval
            vals3= df_fres_posdef[k].to_numpy() - tval
            dx   = np.max(vals) - np.min(vals)
            xmin = np.min(vals) - dx * 0.01; 
            xmax = np.max(vals) + dx * 0.01;
            if 'a0' in k or 'a1' in k: 
                sval = np.sqrt(true_covariance[k.replace('_val', '')][k.replace('_val', '')])
                xmin =-1.8*sval
                xmax = 1.8*sval

            condlm  = np.logical_and(vals > xmin , vals < xmax)
            condlm2 = np.logical_and(vals2 > xmin, vals2 < xmax)
            condlm3 = np.logical_and(vals3 > xmin, vals3 < xmax)
            vals    = vals[condlm]
            vals2   = vals2[condlm2]
            vals3   = vals3[condlm3]
            errmean = np.mean(df_fres_posdef[k.replace('_val', '_err')].to_numpy())

            hst  = TH1D("hval"+k, "hval"+k, 50, xmin, xmax)
            hst.SetXTitle(k)
            hst.SetYTitle("Freq")
            hst.SetLineColor(4)
            hst.SetLineStyle(1)
            hst.SetLineWidth(3)
            for val in vals: hst.Fill(val)
            ymax   = hst.GetMaximum()
            h_val  = np.mean(vals)
            h_sval = np.std(vals)
    
            hst3 = TH1D("h3val"+k, "h3val"+k, 50, xmin, xmax)
            hst3.SetXTitle(k)
            hst3.SetYTitle("Freq")
            hst3.SetLineColor(2)
            hst3.SetLineStyle(9)
            hst3.SetLineWidth(3)
            for val3 in vals3: hst3.Fill(val3)
            hst2 = TH1D("h2val"+k, "h2val"+k, 50, xmin, xmax)
            hst2.SetXTitle(k)
            hst2.SetYTitle("Freq")
            hst2.SetLineColor(8)
            hst2.SetLineStyle(9)
            hst2.SetLineWidth(3)
            for val2 in vals2: hst2.Fill(val2)
    
            #True val
            line_g = TLine(0.,0.,0.,ymax) 
            line_g.SetLineColor(2)
            #Mean val
            line_e = TLine(h_val,0.,h_val,ymax); 
            line_e.SetLineStyle(2)
            line_e.SetLineColor(6); 
            if 'a0' in k or 'a1' in k: 
                #Lattice value
                llow_g  = TLine(-sval,0.,-sval,ymax); llow_g.SetLineColor(4)
                lhigh_g = TLine(+sval,0.,+sval,ymax); lhigh_g.SetLineColor(4)
                #hist value
                llow_e  = TLine(h_val-h_sval,0.,h_val-h_sval,ymax); llow_e.SetLineColor(4);  llow_e.SetLineStyle(2)
                lhigh_e = TLine(h_val+h_sval,0.,h_val+h_sval,ymax); lhigh_e.SetLineColor(4); lhigh_e.SetLineStyle(2)
    
            c2 = TCanvas("c1val"+k,"c1val"+k, 600, 400)
            hst.Draw("")
            hst3.Draw("same")
            hst2.Draw("same")
            line_g.Draw("same")
            line_e.Draw("same")
            #fit    = hst3.Fit("gausn")
            #myfunc = hst3.GetFunction("gausn")
            if 'a0' in k or 'a1' in k: 
                llow_g.Draw("same")
                lhigh_g.Draw("same")
                llow_e.Draw("same")
                lhigh_e.Draw("same")

            diffmean_val[k.replace('_val', '')]  =  (np.mean(vals), np.std(vals)/np.sqrt(len(vals)), np.std(vals)/np.sqrt(len(vals)))
            if 'a0' in k or 'a1' in k: 
                #std dev, std error, lqcd std dev, mean of errors from fit
                #diffsigma_val[k.replace('_val', '')] =  (myfunc.GetParameter(2), sval, errmean)
                diffsigma_val[k.replace('_val', '')] =  (np.std(vals), sval, errmean)
            else:
                #diffsigma_val[k.replace('_val', '')] =  (myfunc.GetParameter(2), myfunc.GetParameter(2), errmean)
                diffsigma_val[k.replace('_val', '')] =  (np.std(vals), np.std(vals), errmean)
    
            pfname = valuedir+'plot_'+k+'_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
            c2.SaveAs(pfname)
        elif 'loglh' in k or '_err' in k:
            vals = df_fres[k].to_numpy()
            vals3= df_fres_posdef[k].to_numpy()
            vals2= df_fres_nonposdef[k].to_numpy()
            dx   = np.max(vals) - np.min(vals)
            xmin = np.min(vals) - dx * 0.01; 
            xmax = np.max(vals) + dx * 0.01;
            condlm  = np.logical_and( vals > xmin , vals < xmax)
            condlm2 = np.logical_and(vals2 > xmin, vals2 < xmax)
            condlm3 = np.logical_and(vals3 > xmin, vals3 < xmax)
            vals    = vals[condlm]
            vals2   = vals2[condlm2]
            vals3   = vals3[condlm3]
    
            hst  = TH1D("hval"+k, "hval"+k, 50, xmin, xmax)
            hst.SetXTitle(k)
            hst.SetYTitle("Freq")
            hst.SetLineColor(4)
            hst.SetLineStyle(1)
            hst.SetLineWidth(3)
            for val in vals: hst.Fill(val)
            ymax   = hst.GetMaximum()
    
            hst3 = TH1D("h3val"+k, "h3val"+k, 50, xmin, xmax)
            hst3.SetXTitle(k)
            hst3.SetYTitle("Freq")
            hst3.SetLineColor(2)
            hst3.SetLineStyle(9)
            hst3.SetLineWidth(3)
            for val3 in vals3: hst3.Fill(val3)
            hst2 = TH1D("h2val"+k, "h2val"+k, 50, xmin, xmax)
            hst2.SetXTitle(k)
            hst2.SetYTitle("Freq")
            hst2.SetLineColor(8)
            hst2.SetLineStyle(9)
            hst2.SetLineWidth(3)
            for val2 in vals2: hst2.Fill(val2)
    
            #True val
            h_val= np.mean(vals)
            true_val[k] = h_val #Should be changed!!!
            line_g = TLine(true_val[k],0.,true_val[k],ymax) 
            line_g.SetLineColor(2)
            #Mean val
            line_e = TLine(h_val,0.,h_val,ymax); 
            line_e.SetLineStyle(2)
            line_e.SetLineColor(6); 
    
            c3 = TCanvas("c1val"+k,"c1val"+k, 600, 400)
            hst.Draw("")
            hst3.Draw("same")
            hst2.Draw("same")
            line_g.Draw("same")
            line_e.Draw("same")
    
            pfname = errdir+'plot_'+k+'_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
            c3.SaveAs(pfname)

    ############ Pull 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Fit {0} toys (Generated {1:.1e} evnts/toy, forced pos. def. {2:.1f}%)'.format(seedmax-seedmin, float(nevnts), 100.*nfrac) , fontsize=10)
    ax.set_ylabel('Pull Mean and Std Dev' , fontsize=15)

    ycenter     = 0.
    fill_mnstd_axis(ax, floatedvars, pullmean_val, pullsigma_val, ycenter)

    fig.tight_layout()
    pfname = basedir+'plot_Pull_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
    fig.savefig(pfname, bbox_inches = 'tight')
    #############

    ############ Pull mean and Std seperately
    fig = plt.figure()
    ax = fig.add_subplot(111)
    empty_axis(ax)
    ax.set_title('Fit {0} toys (Generated {1:.1e} evnts/toy, forced pos. def. {2:.1f}%)'.format(seedmax-seedmin, float(nevnts), 100.*nfrac) , fontsize=10)

    #first axis
    ax1 = fig.add_subplot(211)
    ax1.set_ylabel('Pull Mean' , fontsize=15)
    ycenter     = 0.
    if len(floatedvars) != 1:
        fill_pullaxis(ax1, floatedvars, pullmean_val, ycenter)
    else:
        fill_pullaxis(ax1, floatedvars, pullmean_val, ycenter, xoffset = 0.005)

    #second axis
    ax2 = fig.add_subplot(212)
    ax2.set_ylabel('Pull Std Dev' , fontsize=15)
    ycenter     = 1.
    if len(floatedvars) != 1:
        fill_pullaxis(ax2, floatedvars, pullsigma_val, ycenter)
    else:
        fill_pullaxis(ax2, floatedvars, pullsigma_val, ycenter, xoffset = 0.005)

    fig.tight_layout()
    pfname = basedir+'plot_PullMeanStdDev_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
    fig.savefig(pfname, bbox_inches = 'tight')
    ############

    ############ Value
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Fit {0} toys (Generated {1:.1e} evnts/toy, forced pos. def. {2:.1f}%)'.format(seedmax-seedmin, float(nevnts), 100.*nfrac) , fontsize=10)
    ax.set_ylabel(r'Mean and Std Dev', fontsize=10, labelpad=15)

    fvars1      = [k for k in floatedvars if 'a0' in k or 'C' in k]
    fvars2      = [k for k in floatedvars if k not in fvars1]
    ycenter     = 0.
    if len(fvars2) == 0:
        fill_mnstd_axis(ax, fvars1, diffmean_val, diffsigma_val, ycenter, plottype = 'meanstddev', color = 'b')
    else:
        empty_axis(ax)
        #first axis
        ax1         = fig.add_subplot(211)
        fill_mnstd_axis(ax1, fvars1, diffmean_val, diffsigma_val, ycenter, plottype = 'meanstddev', color = 'b')
        #second axis
        ax2         = fig.add_subplot(212)
        fill_mnstd_axis(ax2, fvars2, diffmean_val, diffsigma_val, ycenter, plottype = 'meanstddev', color = 'b')

    fig.tight_layout()
    pfname = basedir+'plot_Value_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
    fig.savefig(pfname, bbox_inches = 'tight')
    ############

    ############ Value Mean and error on mean
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Fit {0} toys (Generated {1:.1e} evnts/toy, forced pos. def. {2:.1f}%)'.format(seedmax-seedmin, float(nevnts), 100.*nfrac) , fontsize=10)
    ax.set_ylabel(r'Mean and error on mean', fontsize=10, labelpad=15)

    fvars1      = [k for k in floatedvars if 'a0' in k or 'C' in k]
    fvars2      = [k for k in floatedvars if k not in fvars1]
    ycenter     = 0.
    if len(fvars2) == 0:
        if len(floatedvars) != 1:
            fill_pullaxis(ax, fvars1, diffmean_val, ycenter, color = 'b')
        else:
            fill_pullaxis(ax, fvars1, diffmean_val, ycenter, color = 'b', xoffset = 0.005)
    else:
        empty_axis(ax)
        #first axis
        ax1         = fig.add_subplot(211)
        fill_pullaxis(ax1, fvars1, diffmean_val, ycenter, color = 'b')
        #second axis
        ax2         = fig.add_subplot(212)
        fill_pullaxis(ax2, fvars2, diffmean_val, ycenter, color = 'b')

    fig.tight_layout()
    pfname = basedir+'plot_ValueMeanItsError_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
    fig.savefig(pfname, bbox_inches = 'tight')
    #############

    ############ Value width comparision
    ffvars      = [k for k in floatedvars if 'a0' in k or 'a1' in k]
    if len(ffvars) != 0:
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        empty_axis(ax)
        ax.set_title('Fit {0} toys (Generated {1:.1e} evnts/toy, forced pos. def. {2:.1f}%)'.format(seedmax-seedmin, float(nevnts), 100.*nfrac) , fontsize=10)


        #first axis
        ax1         = fig.add_subplot(211)
        ycenter     = 1.
        ax1.set_ylabel(r'Std Dev ratio $\frac{Toys}{LQCD}$', fontsize=10, labelpad=15)
        fill_stddevaxis(ax1, ffvars, diffsigma_val, ycenter, factor = 2.0)

        #second axis
        ax2         = fig.add_subplot(212)
        ax2.set_ylabel(r'Std Dev ratio $\frac{Toys}{Mean\,of\,fit\,errors}$', fontsize=10, labelpad=15)
        fill_stddevaxis(ax2, ffvars, diffsigma_val, ycenter, plottype = 'fit', factor = 1.5)

        fig.tight_layout()
        pfname = basedir+'plot_ValueStdDev_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
        fig.savefig(pfname, bbox_inches = 'tight')
    #############

    ############ Correlation matrix (Fit, LQCD, Toys)
    if len(floatedvars) != 1 and get_covariance:
        df_fitcov, df_toycov = get_covmatrices(floatedvars, df_fres, getcorr = True)
        #print(df_fitcov)
        #print(df_toycov)

        #Toys correlation
        fig, ax = plt.subplots(figsize=(11, 15))
        vmin  = -1.; vmax  = 1.
        ticks = [vmin, 0.5, 0., 0.5, vmax]
        plotcorr(ax, df_toycov, vmin = vmin , vmax = vmax, ticks = ticks, center = 0.)
        pfname = basedir+'plot_ToysCorr_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
        fig.savefig(pfname, bbox_inches = 'tight')

        #Toys/Fit correlation
        fig, ax   = plt.subplots(figsize=(11, 15))
        dftoyOfit = df_toycov/df_fitcov
        #vmin  = np.min(dftoyOfit.to_numpy())
        #vmax  = np.max(dftoyOfit.to_numpy())
        #ticks  = np.linspace(vmin, vmax, 5)
        vmin  = -5.; vmax  =  5.
        ticks  = np.linspace(vmin+1., vmax+1., 5)
        plotcorr(ax, dftoyOfit, vmin = vmin, vmax = vmax, ticks = ticks, center = 1.)
        pfname = basedir+'plot_ToysOFitCorr_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
        fig.savefig(pfname, bbox_inches = 'tight')

        #for form factors only Toys/LQCD
        if len(ffvars) != 0:
            _, df_toycovff, df_truecovff = get_covmatrices(ffvars, df_fres, getcorr = True, getlqcd = True)
            #print(df_toycovff)
            #print(df_truecovff)

            fig, ax   = plt.subplots(figsize=(11, 15))
            dftoyOlqcd= df_toycovff/df_truecovff
            #vmin  = np.min(dftoyOlqcd.to_numpy())
            #vmax  = np.max(dftoyOlqcd.to_numpy())
            ##vmin = -max(np.abs([np.min(dftoyOlqcd.to_numpy()), np.max(dftoyOlqcd.to_numpy())]))
            ##vmax =  max(np.abs([np.min(dftoyOlqcd.to_numpy()), np.max(dftoyOlqcd.to_numpy())]))
            ##ticks = np.linspace(vmin+1., vmax+1., 9)
            #ticks  = np.linspace(vmin, vmax, 5)
            vmin  = -5.; vmax  =  5.
            ticks  = np.linspace(vmin+1., vmax+1., 5)
            plotcorr(ax, dftoyOlqcd, vmin = vmin, vmax = vmax, ticks = ticks, center = 1.)
            pfname = basedir+'plot_ToysOLQCDCorr_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
            fig.savefig(pfname, bbox_inches = 'tight')
    ##########

    #save the data frame as root file
    pfname = direc+'root_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.root'
    to_root(df_fres, pfname, key='tree', store_index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for plot_toy_results.py')
    parser.add_argument('-f'   , '--floatWC'   , dest='floatWC'   ,type=str,  default='None'    , help='(string) Name of the Wilson coefficient (WC) floated when fitting. Available options are [CVR,CSR,CSL,CT,None]. Default is None.')
    parser.add_argument('-smin', '--seedmin'   , dest='seedmin'   ,type=int,  default=int(0)    , help='(int) Min value of seed (an index for the toy ) used in the generation of fake/toy data. Default is zero.')
    parser.add_argument('-smax', '--seedmax'   , dest='seedmax'   ,type=int,  default=int(100)  , help='(int) Max value of seed (an index for the toy ) used in the generation of fake/toy data. Default is 100.')
    parser.add_argument('-b'   , '--bscheme'   , dest='bscheme'   ,type=str,  default='Scheme2' , help='(string) Binning scheme to be used when fitting. Default is Scheme2. See BinningSchemes/Binning_Scheme.py.')
    parser.add_argument('-n'   , '--nevnts'    , dest='nevnts'    ,type=int,  default=int(7.5e6), help='(int) Size of the toy sample generated. Default is 7.5M events.')
    parser.add_argument('-d'   , '--direc'     , dest='direc'     ,type=str,  default='./plots/', help='(string) Directory in which the fit result (.txt) is saved. Default in current directory.')
    parser.add_argument('-e'   , '--floated_FF', dest='floated_FF',nargs='+', default = ['None'], help="(list) List of form factor (FF) parameters floated in fit. See LbToLclnu_fit_binscheme.py. Default is ['None'].") 
    parser.add_argument('-sf'  , '--suffix'     , dest='suffix'    ,type=str,  default='toy'     , help="(int) A unique suffix added to the name of the fit result file (*_suffix.txt). Default is 'toy'.")
    parser.add_argument('-cov', '--get_covariance', dest='get_covariance',type=str2bool,default='False',help='Set to True, if you want to get the covariance matrix. Default is false.')
    args       = parser.parse_args()
    floatWC    = args.floatWC
    seedmin    = args.seedmin
    seedmax    = args.seedmax
    bscheme    = args.bscheme
    nevnts     = args.nevnts
    suffix     = args.suffix
    direc      = args.direc
    floated_FF = args.floated_FF
    get_covariance = args.get_covariance
    print(args)
    if not direc.endswith('/'): direc += '/'

    ######### set the true values used in the generation
    true_val = {}
    #WC values
    true_val["CVL"] = 0.0
    true_val["CVR"] = 0.0
    true_val["CSR"] = 0.0
    true_val[ "CT"] = 0.0
    true_val["CSL"] = 0.0
    if 'NP1' in suffix:
        true_val[floatWC] = 1.0
    elif 'NP2' in suffix:
        true_val[floatWC] =-1.0

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
    pulldir = basedir+'Pull/'
    errdir  = basedir+'Errs/'
    valuedir= basedir+'Value/'
    if not os.path.exists(basedir):  os.system('mkdir '+basedir)
    if not os.path.exists(pulldir):  os.system('mkdir '+pulldir)
    if not os.path.exists(errdir):   os.system('mkdir '+errdir)
    if not os.path.exists(valuedir): os.system('mkdir '+valuedir)

    main()
