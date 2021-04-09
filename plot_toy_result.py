from ROOT import TFile, TTree, TH1D, TH2D, gROOT, gStyle, TCanvas, kGreen, kRed, TCut, TPaveText, TLegend, kBlue, gPad, TLine, TF1, TGraph2D, TF2, TEllipse
gROOT.SetBatch(True)
gStyle.SetOptFit(1)
import numpy as np
import sys, os
import pandas as pd
import argparse
fitdir=os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt

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

def fill_pullaxis(ax1, floatedvars, pullmean_val, ycenter):
    xindx      = np.array(list(range(len(floatedvars))))
    pmcnt      = np.array([pullmean_val[k][0] for k in floatedvars])
    pmerr_lower= np.array([pullmean_val[k][1] for k in floatedvars])
    pmerr_upper= np.array([pullmean_val[k][2] for k in floatedvars])

    ax1.set_ylim(get_ylimits(ycenter, pmcnt, pmerr_lower, pmerr_upper))
    ax1.axhline(y=ycenter, color='black', linestyle='--', alpha = 0.8)
    for xind in xindx: ax1.axvline(x=xind, color='black', linestyle='dotted', alpha = 0.1)
    ax1.errorbar(xindx, pmcnt, yerr=[1.*pmerr_lower, 1.*pmerr_upper], fmt='s', color='r', alpha = 0.8, capsize=4, capthick=1)
    for x, y, yerr in zip(xindx, pmcnt, pmerr_lower): ax1.text(x+0.2, y, r'{0:1.2f}$\sigma$'.format(abs((y-ycenter)/yerr)), fontsize=7, color='r', alpha=0.8)
    ax1.set_xticks(xindx, minor=False)
    ax1.set_xticklabels(floatedvars, fontdict=None, rotation=45, ha='right', minor=False)

def fill_meanaxis(ax2, fvars2, diffmean_val, ycenter, factor = 1.):
    xindx2           = np.array(list(range(len(fvars2))))
    ntoys2           = np.array([diffmean_val[k][0] for k in fvars2])
    dfmcnt2          = np.array([diffmean_val[k][1] for k in fvars2])
    dfmerr_lower2    = np.array([diffmean_val[k][2] for k in fvars2])
    dfmerr_upper2    = np.array([diffmean_val[k][3] for k in fvars2])
    dferrOnm_lower2  = dfmerr_lower2/np.sqrt(ntoys2)
    dferrOnm_upper2  = dfmerr_upper2/np.sqrt(ntoys2)

    ax2.set_ylim(get_ylimits(ycenter, dfmcnt2, dferrOnm_lower2, dferrOnm_upper2, factor = factor))
    ax2.axhline(y=ycenter, color='black', linestyle='--', alpha = 0.8)
    for xind in xindx2: ax2.axvline(x=xind, color='black', linestyle='dotted', alpha = 0.1)
    ax2.errorbar(xindx2, dfmcnt2, yerr=[1.*dferrOnm_lower2, 1.*dferrOnm_upper2], fmt='s', color='b', alpha = 0.8, capsize=4, capthick=1)
    for x, y, yerr in zip(xindx2, dfmcnt2, dferrOnm_lower2): ax2.text(x+0.2, y, r'{0:1.2f}$\sigma$'.format(abs((y-ycenter)/yerr)), fontsize=7, color='b', alpha=0.8)
    ax2.set_xticks(xindx2, minor=False)
    ax2.set_xticklabels(fvars2, fontdict=None, rotation=45, ha='right', minor=False)

def fill_widthaxis(ax2, fvars2, diffmean_val, ycenter, plottype = 'lqcd', factor = 1.):
    xindx2           = np.array(list(range(len(fvars2))))
    dfm_err_lower2    = np.array([diffmean_val[k][2] for k in fvars2])
    if plottype == 'lqcd':
        dfm_other_lower2= np.array([diffmean_val[k][4] for k in fvars2])
    elif plottype == 'fit':
        dfm_other_lower2= np.array([diffmean_val[k][6] for k in fvars2])
    else:
        raise Exception('Plottype not recognised')

    merrother = dfm_err_lower2/dfm_other_lower2
    ax2.set_ylim(get_ylimits(ycenter, merrother, np.zeros_like(merrother), np.zeros_like(merrother), factor = factor))
    ax2.axhline(y=ycenter, color='black', linestyle='--', alpha = 0.8)
    for xind in xindx2: ax2.axvline(x=xind, color='black', linestyle='dotted', alpha = 0.1)
    ax2.errorbar(xindx2, merrother, fmt='s', color='b', alpha = 0.8, capsize=4, capthick=1)
    ax2.set_xticks(xindx2, minor=False)
    ax2.set_xticklabels(fvars2, fontdict=None, rotation=45, ha='right', minor=False)

def main():
    ######### - just to get the headers by reading one of the toy i.e. when seed = 1
    nonparams  = []
    nonparams += ['loglh']
    nonparams += ['is_valid']
    nonparams += ['has_parameters_at_limit']
    nonparams += ['has_accurate_covar']
    nonparams += ['has_posdef_covar']
    nonparams += ['has_made_posdef_covar']

    fres     = {}
    resfname = direc+'results_'+floatWC+'_1_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.txt'
    if not os.path.isfile(resfname): print('File does not exist', resfname)
    f = open(resfname, "r")
    for l in f.readlines(): 
        varname = l.split()[0]
        if varname not in nonparams:
            fres[varname+'_val']  = []
            fres[varname+'_err']  = []
            fres[varname+'_pull'] = []
        else:
            fres[varname]   = []

    f.close()
    #########
    
    #########
    for i in range(seedmin, seedmax+1):
        resfname = direc+'results_'+floatWC+'_'+str(i)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.txt'
        if not os.path.isfile(resfname): 
            print('File does not exist', resfname)
            continue
    
        f = open(resfname, "r")
        ls= f.readlines()
        for l in ls: 
            varname = l.split()[0]
            if varname not in nonparams:
                tval    = trueval[varname]
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
    
    df_fres   = pd.DataFrame.from_dict(fres)
    print('Before requirements of valid minima and params in limit', df_fres.shape[0])
    condvalid     = df_fres['is_valid'] == 1
    condparmlmt   = df_fres['has_parameters_at_limit'] == 0
    cond          = np.logical_and(condvalid, condparmlmt)
    df_fres       = df_fres[cond]
    print('After requirements of valid minima and params in limit', df_fres.shape[0])

    condacccov    = df_fres['has_accurate_covar'] == 1
    condposdefcov = df_fres['has_posdef_covar'] == 1
    condmadeposdef= df_fres['has_made_posdef_covar'] == 0
    cond          = np.logical_and(condacccov, condposdefcov)
    cond          = np.logical_and(cond, condmadeposdef)
    df_res_posdef    = df_fres[ cond]
    df_res_nonposdef = df_fres[np.logical_not(cond)]
    print('After nonposdef requirements', df_res_nonposdef.shape)
    print('After posdef requirements'   , df_res_posdef.shape)
    nfrac   = float(df_res_nonposdef.shape[0])/df_fres.shape[0]

    pullmean_val  = {}
    pullsigma_val = {}
    diffmean_val  = {}
    for k in list(df_fres.keys()):
        if 'pull' in k:
            limitpull = 5.
            condlimit2= np.abs(df_res_nonposdef[k]) < limitpull
            condlimit3= np.abs(df_res_posdef[k]) < limitpull
            vals2     = df_res_nonposdef[condlimit2][k].to_numpy()
            vals3     = df_res_posdef[condlimit3][k].to_numpy()
            emptyval2 = len(vals2) == 0
            emptyval3 = len(vals3) == 0
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
            fit = hst3.Fit("gausn")
            myfunc = hst3.GetFunction("gausn")
            pullmean_val[k.replace('_pull', '')]  = (myfunc.GetParameter(1), myfunc.GetParError(1), myfunc.GetParError(1))
            pullsigma_val[k.replace('_pull', '')] = (myfunc.GetParameter(2), myfunc.GetParError(2), myfunc.GetParError(1))
            pfname = direc+'plots/Pull/plot_'+k+'_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
            c1.SaveAs(pfname)
        elif '_val' in k and 'is_valid' not in k:
            #print(k)
            vals = df_fres[k].to_numpy()   
            vals2= df_res_nonposdef[k].to_numpy()
            vals3= df_res_posdef[k].to_numpy()
            dx   = np.max(vals) - np.min(vals)
            xmin = np.min(vals) - dx * 0.01; 
            xmax = np.max(vals) + dx * 0.01;
            tval = trueval[k.replace('_val', '')]
            if 'a0' in k or 'a1' in k: 
                sval = np.sqrt(truecovariance[k.replace('_val', '')][k.replace('_val', '')])
                xmin = tval - 1.8*sval
                xmax = tval + 1.8*sval

            condlm  = np.logical_and(vals > xmin , vals < xmax)
            condlm2 = np.logical_and(vals2 > xmin, vals2 < xmax)
            condlm3 = np.logical_and(vals3 > xmin, vals3 < xmax)
            vals    = vals[condlm]
            vals2   = vals2[condlm2]
            vals3   = vals3[condlm3]
            diffvals= tval - vals3
            errmean = np.mean(df_res_posdef[k.replace('_val', '_err')].to_numpy())
            if 'a0' in k or 'a1' in k: 
                diffmean_val[k.replace('_val', '')]  =  (len(diffvals), np.mean(diffvals), np.std(diffvals), np.std(diffvals), sval, sval, errmean, errmean)
            else:
                diffmean_val[k.replace('_val', '')]  =  (len(diffvals), np.mean(diffvals), np.std(diffvals), np.std(diffvals), np.std(diffvals), np.std(diffvals), errmean, errmean)
    
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
            line_g = TLine(trueval[k.replace('_val', '')],0.,trueval[k.replace('_val', '')],ymax) 
            line_g.SetLineColor(2)
            #Mean val
            line_e = TLine(h_val,0.,h_val,ymax); 
            line_e.SetLineStyle(2)
            line_e.SetLineColor(6); 
            if 'a0' in k or 'a1' in k: 
                #Lattice value
                llow_g  = TLine(tval-sval,0.,tval-sval,ymax); llow_g.SetLineColor(4)
                lhigh_g = TLine(tval+sval,0.,tval+sval,ymax); lhigh_g.SetLineColor(4)
                #hist value
                llow_e  = TLine(h_val-h_sval,0.,h_val-h_sval,ymax); llow_e.SetLineColor(4);  llow_e.SetLineStyle(2)
                lhigh_e = TLine(h_val+h_sval,0.,h_val+h_sval,ymax); lhigh_e.SetLineColor(4); lhigh_e.SetLineStyle(2)
    
            c2 = TCanvas("c1val"+k,"c1val"+k, 600, 400)
            hst.Draw("")
            hst3.Draw("same")
            hst2.Draw("same")
            line_g.Draw("same")
            line_e.Draw("same")
            if 'a0' in k or 'a1' in k: 
                llow_g.Draw("same")
                lhigh_g.Draw("same")
                llow_e.Draw("same")
                lhigh_e.Draw("same")
    
            pfname = direc+'plots/Value/plot_'+k+'_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
            c2.SaveAs(pfname)
        elif 'loglh' in k or '_err' in k:
            vals = df_fres[k].to_numpy()
            vals3= df_res_posdef[k].to_numpy()
            vals2= df_res_nonposdef[k].to_numpy()
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
            trueval[k] = h_val #Should be changed!!!
            line_g = TLine(trueval[k],0.,trueval[k],ymax) 
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
    
            pfname = direc+'plots/Errs/plot_'+k+'_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
            c3.SaveAs(pfname)

    floatedvars= list(pullmean_val.keys())

    ############ Pull mean and sigma values
    fig = plt.figure()
    ax = fig.add_subplot(111)
    empty_axis(ax)
    ax.set_title('Fit {0} toys (Generated {1:.1e} evnts/toy, forced pos. def. {2:.1f}%)'.format(seedmax-seedmin, float(nevnts), 100.*nfrac) , fontsize=10)

    #first axis
    ax1 = fig.add_subplot(211)
    ax1.set_ylabel('Pull Mean' , fontsize=15)
    ycenter     = 0.
    fill_pullaxis(ax1, floatedvars, pullmean_val, ycenter)

    #second axis
    ax2 = fig.add_subplot(212)
    ax2.set_ylabel('Pull Width' , fontsize=15)
    ycenter     = 1.
    fill_pullaxis(ax2, floatedvars, pullsigma_val, ycenter)

    fig.tight_layout()
    pfname = direc+'plots/plot_PullMeanWidths_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
    fig.savefig(pfname, bbox_inches = 'tight')
    ############

    ############ Mean and error on mean
    fig = plt.figure()
    ax = fig.add_subplot(111)
    empty_axis(ax)
    ax.set_title('Fit {0} toys (Generated {1:.1e} evnts/toy, forced pos. def. {2:.1f}%)'.format(seedmax-seedmin, float(nevnts), 100.*nfrac) , fontsize=10)
    ax.set_ylabel(r'Mean and error on mean', fontsize=10, labelpad=15)

    #first axis
    ax1         = fig.add_subplot(211)
    fvars1      = [k for k in floatedvars if 'a0' in k or 'C' in k]
    ycenter     = 0.
    fill_meanaxis(ax1, fvars1, diffmean_val, ycenter)

    #second axis
    ax2         = fig.add_subplot(212)
    fvars2      = [k for k in floatedvars if 'a1' in k]
    ycenter     = 0.
    fill_meanaxis(ax2, fvars2, diffmean_val, ycenter, factor = 1.5)

    fig.tight_layout()
    pfname = direc+'plots/plot_Mean_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
    fig.savefig(pfname, bbox_inches = 'tight')
    #############

    ############ width comparision
    fig = plt.figure()
    ax = fig.add_subplot(111)
    empty_axis(ax)
    ax.set_title('Fit {0} toys (Generated {1:.1e} evnts/toy, forced pos. def. {2:.1f}%)'.format(seedmax-seedmin, float(nevnts), 100.*nfrac) , fontsize=10)

    #first axis
    ax1         = fig.add_subplot(211)
    ycenter     = 1.
    ax1.set_ylabel(r'$\frac{1\sigma\,Interval\,(Toys)}{1\sigma\,Interval\,(LQCD)}$', fontsize=10, labelpad=15)
    fill_widthaxis(ax1, floatedvars, diffmean_val, ycenter, factor = 2.0)

    #second axis
    ax2         = fig.add_subplot(212)
    ycenter     = 1.
    ax2.set_ylabel(r'$\frac{1\sigma\,Interval\,(Toys)}{(Mean\,of\,fit\,error)}$', fontsize=10, labelpad=15)
    fill_widthaxis(ax2, floatedvars, diffmean_val, ycenter, plottype = 'fit', factor = 1.5)

    fig.tight_layout()
    pfname = direc+'plots/plot_Widths_'+floatWC+'_'+str(seedmin)+'_'+str(seedmax)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix+'.pdf'
    fig.savefig(pfname, bbox_inches = 'tight')
    #############

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    ## Turn off axis lines and ticks of the big subplot
    #ax.spines['top'].set_color('none')
    #ax.spines['bottom'].set_color('none')
    #ax.spines['left'].set_color('none')
    #ax.spines['right'].set_color('none')
    #ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    #ax.set_title('Fit {0} toys (Generated {1:.1e} evnts/toy, forced pos. def. {2:.1f}%)'.format(seedmax-seedmin, float(nevnts), 100.*nfrac) , fontsize=10)
    #ax.set_ylabel(r'Mean and width of distribution', fontsize=10, labelpad=15)
    #ax.set_xlabel('Parameters', fontsize=10, labelpad=15)
    #ax1 = fig.add_subplot(211)
    #ax2 = fig.add_subplot(212)
    #fvars1           = [k for k in floatedvars if 'a0' in k or 'C' in k]
    #xindx1           = np.array(list(range(len(fvars1))))
    #dfmcnt1          = np.array([diffmean_val[k][0] for k in fvars1])
    #dfmerr_lower1    = np.array([diffmean_val[k][1] for k in fvars1])
    #dfmerr_upper1    = np.array([diffmean_val[k][2] for k in fvars1])
    #dfmlqcderr_lower1= np.array([diffmean_val[k][3] for k in fvars1])
    #dfmlqcderr_upper1= np.array([diffmean_val[k][4] for k in fvars1])
    #dfmfiterr_lower1 = np.array([diffmean_val[k][5] for k in fvars1])
    #dfmfiterr_upper1 = np.array([diffmean_val[k][6] for k in fvars1])
    #fvars2           = [k for k in floatedvars if 'a1' in k]
    #xindx2           = np.array(list(range(len(fvars2))))
    #dfmcnt2          = np.array([diffmean_val[k][0] for k in fvars2])
    #dfmerr_lower2    = np.array([diffmean_val[k][1] for k in fvars2])
    #dfmerr_upper2    = np.array([diffmean_val[k][2] for k in fvars2])
    #dfmlqcderr_lower2= np.array([diffmean_val[k][3] for k in fvars2])
    #dfmlqcderr_upper2= np.array([diffmean_val[k][4] for k in fvars2])
    #dfmfiterr_lower2 = np.array([diffmean_val[k][5] for k in fvars2])
    #dfmfiterr_upper2 = np.array([diffmean_val[k][6] for k in fvars2])
    #ax1.axhline(y=0.0, color='black', linestyle='--', alpha = 0.8)
    #ax1.errorbar(xindx1, dfmcnt1, yerr=[1.*dfmlqcderr_lower1, 1.*dfmlqcderr_upper1], fmt='s', color='g', alpha = 0.8, capsize=4, capthick=1, label='LQCD')
    #ax1.errorbar(xindx1, dfmcnt1, yerr=[1.*dfmfiterr_lower1, 1.*dfmfiterr_upper1], fmt='s', color='b', alpha = 0.8, capsize=4, capthick=1, label='Fit') #Fit Err. Mean
    #ax1.errorbar(xindx1, dfmcnt1, yerr=[1.*dfmerr_lower1, 1.*dfmerr_upper1], fmt='s', color='r', alpha = 0.8, capsize=4, capthick=1, label='Toys') #Toys Std. Dev.
    #ax1.set_xticks(xindx1, minor=False)
    #ax1.set_xticklabels(fvars1, fontdict=None, rotation=45, ha='right', minor=False)
    #ax1.legend(bbox_to_anchor=(1.1, 1.05))
    #ax2.axhline(y=0.0, color='black', linestyle='--', alpha = 0.8)
    #ax2.errorbar(xindx2, dfmcnt2, yerr=[1.*dfmlqcderr_lower2, 1.*dfmlqcderr_upper2], fmt='s', color='g', alpha = 0.8, capsize=4, capthick=1)
    #ax2.errorbar(xindx2, dfmcnt2, yerr=[1.*dfmfiterr_lower2, 1.*dfmfiterr_upper2], fmt='s', color='b', alpha = 0.8, capsize=4, capthick=1)
    #ax2.errorbar(xindx2, dfmcnt2, yerr=[1.*dfmerr_lower2, 1.*dfmerr_upper2], fmt='s', color='r', alpha = 0.8, capsize=4, capthick=1)
    #ax2.set_xticks(xindx2, minor=False)
    #ax2.set_xticklabels(fvars2, fontdict=None, rotation=45, ha='right', minor=False)
    #fig.tight_layout()
    #fig.savefig('test3.pdf', bbox_inches = 'tight')

    #import seaborn as sns
    #from matplotlib.patches import Ellipse
    ##mpl.rcParams.update({'font.size': 19})
    ##plt.rcParams.update({'figure.max_open_warning': 0})
    ##sns.set(style="white")
    ##sns.set(font_scale=1.2)
    #fig, ax = plt.subplots()
    #sns_plot = sns.PairGrid(df, vars = i, palette=sns.color_palette("muted"))
    #sns_plot.map_diag(plt.hist, bins = 40)
    #sns_plot.map_lower(plt.scatter)
    #for j in range(len(i)):
    #    for k in range(len(i)):
    #        xlabel = sns_plot.axes[j,k].get_xlabel()
    #        ylabel = sns_plot.axes[j,k].get_ylabel()
    #        #For plotting mean and covariance - Uncomment later
    #        if j == k:
    #            mu, sigma = df[i[j]].mean(), df[i[j]].std()
    #            sns_plot.axes[j,j].annotate('$\mu={0:.2g},\sigma={1:.3f}$'.format(mu, sigma), xy=(0.40, 0.85), xycoords='axes fraction')
    #        elif j != k and k < j:
    #            #mu_k, sigma_k = df[i[k]].mean(), (df[i[k]].std()/np.sqrt(df.shape[0])); 
    #            mu_j, sigma_j = df[i[j]].mean(), df[i[j]].std()
    #            mu_k, sigma_k = df[i[k]].mean(), df[i[k]].std()
    #            points = df.as_matrix(columns = [i[k], i[j]])
    #            plot_point_cov(points, nstd=1, alpha=0.35, color=color_mean, ax = sns_plot.axes[j,k])
    #            #plot_point_cov(points, nstd=2, alpha=0.15, color=color_mean, ax = sns_plot.axes[j,k])
    #            #sns_plot.axes[j,k].axvspan(mu_k-sigma_k, mu_k+sigma_k, alpha=0.10, color=color_mean)
    #            #sns_plot.axes[j,k].axhspan(mu_j-sigma_j, mu_j+sigma_j, alpha=0.10, color=color_mean)
    #            sns_plot.axes[j,k].scatter(mu_k, mu_j, s=48 ,c=color_mean)
    #            sns_plot.axes[j,k].scatter(Trueval[i[k]][Trueval.index.values.tolist()[0]], Trueval[i[j]][Trueval.index.values.tolist()[0]], s=48 ,c=color_trueval)
    #            rho = np.corrcoef(points.T)[0,1]
    #            #sns_plot.axes[j,k].annotate(r'$\rho=${0:.2g}'.format(rho), fontsize = 9, xy=(0.57, 0.85), xycoords='axes fraction')
    #            sns_plot.axes[j,k].annotate(r'$\rho=${0:.2g}'.format(rho), xy=(0.57, 0.85), xycoords='axes fraction')
    #plt.savefig('test3.pdf')

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
    args       = parser.parse_args()
    floatWC    = args.floatWC
    seedmin    = args.seedmin
    seedmax    = args.seedmax
    bscheme    = args.bscheme
    nevnts     = args.nevnts
    suffix     = args.suffix
    direc      = args.direc
    floated_FF = args.floated_FF
    print(args)
    if not direc.endswith('/'): direc += '/'

    ######### set the true values used in the generation
    trueval = {}
    #WC values
    trueval["CVL"] = 0.0
    trueval["CVR"] = 0.0
    trueval["CSR"] = 0.0
    trueval["CSL"] = 0.0
    trueval[ "CT"] = 0.0
    #FF mean values
    f = open(fitdir+'/FF_cov/LambdabLambdac_results.dat', 'r')
    for l in f.readlines(): trueval[l.split()[0]] = float(l.split()[1])
    f.close()
    #trueval['a0gperp']      = trueval['a0gplus']
    #trueval['a0htildeperp'] = trueval['a0htildeplus']
    #print(trueval)
    #FF covariance matrix
    truecovariance = {}
    for n in list(trueval.keys()):
        if ('a0' in n) or ('a1' in n): truecovariance[n] = {}
    
    for l in open(fitdir+"/FF_cov/LambdabLambdac_covariance.dat", "r").readlines(): 
        truecovariance[l.split()[0]][l.split()[1]] = float(l.split()[2])
    
    #print(truecovariance)
    #########

    main()
