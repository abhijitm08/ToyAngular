#!/bin/python

import sys, os
home = os.getenv('HOME')
from ROOT import TH2D, TCanvas, TFile, gROOT, gStyle
gROOT.SetBatch(True)
from root_pandas import read_root
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pickle
basedir = os.path.abspath(os.getcwd())+'/..'
sys.path.append(basedir)
from BinningSchemes.Binning_Scheme import defing_binning_scheme

bin_scheme= 'Scheme5'
BinScheme = defing_binning_scheme()
q2edges   = BinScheme[bin_scheme]['qsq']
cthledges = BinScheme[bin_scheme]['cthl']
q2nbins   = len(BinScheme[bin_scheme]['qsq'])  - 1
cthlnbins = len(BinScheme[bin_scheme]['cthl']) - 1

def fill_hists(hden, hnum, hw, d_den, d_num, hdenscale = None, hnumscale = None, effscale = None):
    print("Filling histograms")

    #den
    for x,y,w in zip(d_den[0], d_den[1], d_den[2]): hden.Fill(x,y,w)
    hden.Sumw2(); 
    if hdenscale is not None: hden.Scale(hdenscale/hden.Integral()); 
    print(hden.GetName(), hden.GetXaxis().GetNbins() * hden.GetYaxis().GetNbins(), hden.Integral())
    #c1 = TCanvas("c1", "c1")
    #hden.Draw("colz")
    #c1.SaveAs("eff_rootfiles/"+hden.GetName()+".pdf")

    #num
    for x,y,w in zip(d_num[0], d_num[1], d_num[2]): hnum.Fill(x,y,w)
    hnum.Sumw2()
    if hnumscale is not None: hnum.Scale(hnumscale/hnum.Integral()); 
    print(hnum.GetName(), hnum.GetXaxis().GetNbins() * hnum.GetYaxis().GetNbins(), hnum.Integral())
    #c2 = TCanvas("c2", "c2")
    #hnum.Draw("colz")
    #c2.SaveAs("eff_rootfiles/"+hnum.GetName()+".pdf")

    #eff
    hw.Sumw2()
    x_nbins = hw.GetXaxis().GetNbins()
    y_nbins = hw.GetYaxis().GetNbins()
    for i in range(0,x_nbins+2):
        for j in range(0,y_nbins+2):
            global_bin_2D = hw.GetBin(i,j)
            k = hnum.GetBinContent(global_bin_2D)
            N = hden.GetBinContent(global_bin_2D)
            eff = 0.; eff_err = 0.
            if N != 0.:
                eff = k/N
                eff_err = 1./N * np.sqrt(k * (1. - k/N))

            #print(k, N)
            #print(global_bin_2D, eff, eff_err)
            hw.SetBinContent(global_bin_2D, eff)
            hw.SetBinError(global_bin_2D, eff_err)

    if effscale is not None: hw.Scale(effscale); 
    c3 = TCanvas("c3", "c3")
    hw.DrawNormalized("colz text")
    if conservative:
        c3.SaveAs("eff_rootfiles/"+hw.GetName()+"_conservative.pdf")
    else:
        c3.SaveAs("eff_rootfiles/"+hw.GetName()+".pdf")

    print('EFFICIENCY===',hw.GetName(), (hw.GetXaxis().GetNbins() * hw.GetYaxis().GetNbins()), hw.Integral()/(hw.GetXaxis().GetNbins() * hw.GetYaxis().GetNbins()))
    heffavg = hw.Integral()/(hw.GetXaxis().GetNbins() * hw.GetYaxis().GetNbins())
    heffmax = hw.GetMaximum()
    heffmin = hw.GetMinimum()
    print(heffmin, heffmax, heffavg)
    print(hw.GetName(), 'Variation of eff is :: {0:.1f} %'.format(((heffmax - heffmin)/heffavg)*100.))
    return None

def get_tot_eff():
    print("Getting Tot eff")
    store_root= False
    #den: nocuts
    files_den  = [basedir+'/model_dependency/model_dependency_rootfiles/LcMuNu_gen_new_SM_modeldependency.root']

    tree_den   = 'DecayTree'
    cut_den    = ''
    colums_den = ['Lb_True_Q2_mu' , 'Lb_True_Costhetal_mu', 'Event_LbProdcorr', 'Event_FFcorr']
    dfgeom_den = read_root(files_den, key=tree_den, where=cut_den, columns=colums_den) 
    q2_den     = dfgeom_den['Lb_True_Q2_mu'].to_numpy()/1e6
    cthl_den   = dfgeom_den['Lb_True_Costhetal_mu'].to_numpy()
    weights_den= (dfgeom_den['Event_FFcorr']*dfgeom_den['Event_LbProdcorr']).to_numpy()
    d_den      = (q2_den, cthl_den, weights_den)
    geomeff    = 0.10120184694165923 #obtained from MC (NB: also the relative shape of efficiency matters not absolute value)
    print("WARNING Hardcoded geomeff", geomeff)
    filteff       = 0.5 * (0.0959052736468 + 0.0960540773744)
    evts_filtgeom = 0.5 * (1257873. + 1271052.)
    scale_den     = evts_filtgeom/filteff/geomeff
    print('Before', dfgeom_den.shape, scale_den)
    print(dfgeom_den)
    del dfgeom_den

    #num: fullselection
    files_num  = [basedir+'/model_dependency/model_dependency_rootfiles/Lb2Lcmunu_MagDown_2016_Combine_SM_modeldependency.root']
    files_num += [basedir+'/model_dependency/model_dependency_rootfiles/Lb2Lcmunu_MagUp_2016_Combine_SM_modeldependency.root']

    tree_num   = 'DecayTree'
    cut_num    = 'isFullsel==1'
    colums_num = ['Lb_True_Q2_mu' , 'Lb_True_Costhetal_mu', 'Event_LbProdcorr', 'Event_TrackCalibcorr', 'Event_PIDCalibEffWeight', 'Event_L0Muoncorr', 'Event_FFcorr']
    dfgeom_num = read_root(files_num, key=tree_num, where=cut_num, columns=colums_num) 
    q2_num     = dfgeom_num['Lb_True_Q2_mu'].to_numpy()/1e6
    cthl_num   = dfgeom_num['Lb_True_Costhetal_mu'].to_numpy()
    weights_num= (dfgeom_num['Event_FFcorr']*dfgeom_num['Event_LbProdcorr']*dfgeom_num['Event_TrackCalibcorr']\
                        *dfgeom_num['Event_PIDCalibEffWeight']*dfgeom_num['Event_L0Muoncorr']).to_numpy()
    d_num      = (q2_num, cthl_num, weights_num)
    print('After', dfgeom_num.shape)
    print(dfgeom_num)
    del dfgeom_num

    if store_root:
        #make file
        f = TFile.Open("./eff_rootfiles/Eff_Tot_SM_nominal.root", "recreate")
        f.cd()

    #make histograms
    h_den   = TH2D("hTot_SM_nominal_den", "hTot_SM_nominal_den;q^{2}[GeV^{2}];cos(#theta_{l});N"          , q2nbins, q2edges, cthlnbins, cthledges)
    h_num   = TH2D("hTot_SM_nominal_num", "hTot_SM_nominal_num;q^{2}[GeV^{2}];cos(#theta_{l});N(GSRFPTMI)", q2nbins, q2edges, cthlnbins, cthledges)
    h_eff   = TH2D("hTot_SM_nominal_eff", "hTot_SM_nominal_eff;q^{2}[GeV^{2}];cos(#theta_{l});Eff(Tot)"   , q2nbins, q2edges, cthlnbins, cthledges)

    #fill histograms
    fill_hists(h_den, h_num, h_eff,  d_den, d_num, hdenscale = scale_den)

    if store_root:
        h_den.Write()
        h_num.Write()
        h_eff.Write()

    #eff
    h_eff.Sumw2()
    x_nbins = h_eff.GetXaxis().GetNbins()
    y_nbins = h_eff.GetYaxis().GetNbins()
    Eff     = np.zeros(shape=(x_nbins,y_nbins))
    for i in range(1,x_nbins+1): #q2[GeV]
        for j in range(1,y_nbins+1): #cthl
            global_bin_2D = h_eff.GetBin(i,j)
            eff    = h_eff.GetBinContent(global_bin_2D)
            Eff[i-1][j-1]    = eff 
    
    #h_eff.Print("all")
    #print(Eff)
    pickle.dump( Eff, open( './Eff_Tot_SM_nominal.p', "wb" ) )
    if store_root:
        f.Close()

    return None

def get_tot_eff_alternate_model(scenario, model_indx, conservative):
    print("Getting Tot eff")
    store_root= False
    #den: nocuts
    if conservative:
        files_den  = [basedir+'/model_dependency/model_dependency_rootfiles_conservative/LcMuNu_gen_new_'+scenario+'_modeldependency.root']
    else:
        files_den  = [basedir+'/model_dependency/model_dependency_rootfiles/LcMuNu_gen_new_'+scenario+'_modeldependency.root']

    tree_den   = 'DecayTree'
    cut_den    = ''
    colums_den = ['Lb_True_Q2_mu' , 'Lb_True_Costhetal_mu', 'Event_LbProdcorr', 'Event_Model_'+model_indx]
    dfgeom_den = read_root(files_den, key=tree_den, where=cut_den, columns=colums_den) 
    q2_den     = dfgeom_den['Lb_True_Q2_mu'].to_numpy()/1e6
    cthl_den   = dfgeom_den['Lb_True_Costhetal_mu'].to_numpy()
    weights_den= (dfgeom_den['Event_Model_'+model_indx]*dfgeom_den['Event_LbProdcorr']).to_numpy()
    d_den      = (q2_den, cthl_den, weights_den)
    geomeff    = 0.10120184694165923 #obtained from MC (NB: also the relative shape of efficiency matters not absolute value)
    print("WARNING Hardcoded geomeff", geomeff)
    filteff       = 0.5 * (0.0959052736468 + 0.0960540773744)
    evts_filtgeom = 0.5 * (1257873. + 1271052.)
    scale_den     = evts_filtgeom/filteff/geomeff
    print('Before', dfgeom_den.shape, scale_den)
    print(dfgeom_den)
    del dfgeom_den

    #num: fullselection
    if conservative:
        files_num  = [basedir+'/model_dependency/model_dependency_rootfiles_conservative/Lb2Lcmunu_MagDown_2016_Combine_'+scenario+'_modeldependency.root']
        files_num += [basedir+'/model_dependency/model_dependency_rootfiles_conservative/Lb2Lcmunu_MagUp_2016_Combine_'+scenario+'_modeldependency.root']
    else:
        files_num  = [basedir+'/model_dependency/model_dependency_rootfiles/Lb2Lcmunu_MagDown_2016_Combine_'+scenario+'_modeldependency.root']
        files_num += [basedir+'/model_dependency/model_dependency_rootfiles/Lb2Lcmunu_MagUp_2016_Combine_'+scenario+'_modeldependency.root']

    tree_num   = 'DecayTree'
    cut_num    = 'isFullsel==1'
    colums_num = ['Lb_True_Q2_mu' , 'Lb_True_Costhetal_mu', 'Event_LbProdcorr', 'Event_TrackCalibcorr', 'Event_PIDCalibEffWeight', 'Event_L0Muoncorr', 'Event_Model_'+model_indx]
    dfgeom_num = read_root(files_num, key=tree_num, where=cut_num, columns=colums_num) 
    q2_num     = dfgeom_num['Lb_True_Q2_mu'].to_numpy()/1e6
    cthl_num   = dfgeom_num['Lb_True_Costhetal_mu'].to_numpy()
    weights_num= (dfgeom_num['Event_Model_'+model_indx]*dfgeom_num['Event_LbProdcorr']*dfgeom_num['Event_TrackCalibcorr']\
                        *dfgeom_num['Event_PIDCalibEffWeight']*dfgeom_num['Event_L0Muoncorr']).to_numpy()
    d_num      = (q2_num, cthl_num, weights_num)
    print('After', dfgeom_num.shape)
    print(dfgeom_num)
    del dfgeom_num

    if store_root:
        #make file
        if conservative:
            f = TFile.Open("./eff_rootfiles/Eff_Tot_"+scenario+"_"+model_indx+"_conservative.root", "recreate")
        else:
            f = TFile.Open("./eff_rootfiles/Eff_Tot_"+scenario+"_"+model_indx+".root", "recreate")

        f.cd()

    #make histograms
    h_den   = TH2D("hTot_"+scenario+"_"+model_indx+"_den", "hTot_"+scenario+"_"+model_indx+"_den;q^{2}[GeV^{2}];cos(#theta_{l});N"          , q2nbins, q2edges, cthlnbins, cthledges)
    h_num   = TH2D("hTot_"+scenario+"_"+model_indx+"_num", "hTot_"+scenario+"_"+model_indx+"_num;q^{2}[GeV^{2}];cos(#theta_{l});N(GSRFPTMI)", q2nbins, q2edges, cthlnbins, cthledges)
    h_eff   = TH2D("hTot_"+scenario+"_"+model_indx+"_eff", "hTot_"+scenario+"_"+model_indx+"_eff;q^{2}[GeV^{2}];cos(#theta_{l});Eff(Tot)"   , q2nbins, q2edges, cthlnbins, cthledges)

    #fill histograms
    fill_hists(h_den, h_num, h_eff,  d_den, d_num, hdenscale = scale_den)

    if store_root:
        h_den.Write()
        h_num.Write()
        h_eff.Write()

    #eff
    h_eff.Sumw2()
    x_nbins = h_eff.GetXaxis().GetNbins()
    y_nbins = h_eff.GetYaxis().GetNbins()
    Eff     = np.zeros(shape=(x_nbins,y_nbins))
    EffErr  = np.zeros(shape=(x_nbins,y_nbins))
    for i in range(1,x_nbins+1): #q2[GeV]
        for j in range(1,y_nbins+1): #cthl
            global_bin_2D = h_eff.GetBin(i,j)
            eff    = h_eff.GetBinContent(global_bin_2D)
            efferr = h_eff.GetBinError(global_bin_2D)
            Eff[i-1][j-1]    = eff 
            EffErr[i-1][j-1] = efferr
    
    #h_eff.Print("all")
    #print(Eff)
    if conservative:
        dirstore = './eff_pickled_conservative'
    else:
        dirstore = './eff_pickled'

    pickle.dump( Eff, open( dirstore+'/Eff_Tot_'+scenario+'_'+model_indx+'.p', "wb" ) )

    if store_root:
        f.Close()

    return None


def main():
    ##Both conservative and not conservative will give same efficiency shapes for SM
    get_tot_eff()
    #get_tot_eff_alternate_model(scenario, model_indx, conservative)

if __name__ == '__main__':
    scenario   = sys.argv[1]
    model_indx = sys.argv[2]
    variation_range   = sys.argv[3]

    conservative = None
    if variation_range == 'large':
        conservative = True
    elif variation_range == 'one_sigma':
        conservative = False
    else:
        raise Exception("The value of variation_range not recognised, only 'large' or 'one_sigma' allowed!")

    print(scenario, model_indx, variation_range, conservative)

    scenarios = ['CVR', 'CSR', 'CSL', 'CT', 'SM']
    if scenario not in scenarios:
        raise Exception('Scenario not in Scenarios')

    gROOT.ProcessLine(".x lhcbStyle2D.C")
    gStyle.SetPaintTextFormat("1.3f")
    main()
