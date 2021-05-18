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
sys.path.append(os.path.abspath(os.getcwd())+'/../')
from BinningSchemes.Binning_Scheme import defing_binning_scheme

mcdir = '/disk/lhcb_data/amathad/Lb2Lclnu_analysis/MC'

bin_scheme= 'Scheme5'
BinScheme = defing_binning_scheme()
q2edges   = BinScheme[bin_scheme]['qsq']
cthledges = BinScheme[bin_scheme]['cthl']
q2nbins   = len(BinScheme[bin_scheme]['qsq'])  - 1
cthlnbins = len(BinScheme[bin_scheme]['cthl']) - 1

def Setbincontents_to_zero(hist):
    x_nbins = hist.GetXaxis().GetNbins()
    y_nbins = hist.GetYaxis().GetNbins()
    for i in range(0,x_nbins+2):
        for j in range(0,y_nbins+2):
            global_bin_2D = hist.GetBin(i,j)
            hist.SetBinContent(global_bin_2D, 0.)
            hist.SetBinError(global_bin_2D, 0.)

    hist.Sumw2(); 
    #hist.Print('all')
    return None

def fill_hists(hden, hnum, hw, d_den, d_num, hdenscale = None, hnumscale = None, effscale = None):
    print("Filling histograms")

    #den
    for x,y,w in zip(d_den[0], d_den[1], d_den[2]): hden.Fill(x,y,w)
    hden.Sumw2(); 
    if hdenscale is not None: hden.Scale(hdenscale/hden.Integral()); 
    print(hden.GetName(), hden.GetXaxis().GetNbins() * hden.GetYaxis().GetNbins(), hden.Integral())
    #c1 = TCanvas("c1", "c1")
    #hden.Draw("colz")
    #c1.SaveAs("Effs/plots/"+hden.GetName()+".pdf")

    #num
    for x,y,w in zip(d_num[0], d_num[1], d_num[2]): hnum.Fill(x,y,w)
    hnum.Sumw2()
    if hnumscale is not None: hnum.Scale(hnumscale/hnum.Integral()); 
    print(hnum.GetName(), hnum.GetXaxis().GetNbins() * hnum.GetYaxis().GetNbins(), hnum.Integral())
    #c2 = TCanvas("c2", "c2")
    #hnum.Draw("colz")
    #c2.SaveAs("Effs/plots/"+hnum.GetName()+".pdf")

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
    #c3 = TCanvas("c3", "c3")
    #hw.Draw("colz text")
    #c3.SaveAs("Effs/plots/"+hw.GetName()+".pdf")
    print('EFFICIENCY===',hw.GetName(), (hw.GetXaxis().GetNbins() * hw.GetYaxis().GetNbins()), hw.Integral()/(hw.GetXaxis().GetNbins() * hw.GetYaxis().GetNbins()))
    heffavg = hw.Integral()/(hw.GetXaxis().GetNbins() * hw.GetYaxis().GetNbins())
    heffmax = hw.GetMaximum()
    heffmin = hw.GetMinimum()
    print(heffmin, heffmax, heffavg)
    print(hw.GetName(), 'Variation of eff is :: {0:.1f} %'.format(((heffmax - heffmin)/heffavg)*100.))
    return None

def GetTotEff():
    print("Getting Tot eff")

    #den: nocuts
    files_den  = [home+'/LbToLclnu_RunTwo/FittingScripts/qsq_cthl_spectra/Differential_density/responsematrix_eff/GeomEffFiles/LcMuNu_gen_new_'+model_indx+'.root']
    tree_den   = 'DecayTree'
    cut_den    = ''
    colums_den = ['Event_SM', 'Event_LbProdcorr', 'Lb_True_Q2_mu' , 'Lb_True_Costhetal_mu', 'Event_Model_'+model_indx]
    dfgeom_den = read_root(files_den, key=tree_den, where=cut_den, columns=colums_den) 
    q2_den     = dfgeom_den['Lb_True_Q2_mu'].to_numpy()/1e6
    cthl_den   = dfgeom_den['Lb_True_Costhetal_mu'].to_numpy()
    weights_den= (dfgeom_den['Event_SM']*dfgeom_den['Event_LbProdcorr']).to_numpy()
    d_den      = (q2_den, cthl_den, weights_den)
    geomeff    = 0.10120184694165923 #obtained from MC (NB: also the relative shape of efficiency matters not absolute value)
    print("WARNING Hardcoded geomeff", geomeff)
    filteff       = 0.5 * (0.0959052736468 + 0.0960540773744)
    evts_filtgeom = 0.5 * (1257873. + 1271052.)
    scale_den     = evts_filtgeom/filteff/geomeff
    print('Before', dfgeom_den.shape, scale_den)

    #num: fullselection
    files_num  = [mcdir+'/Lb2Lcmunu_MagDown_2016_Combine_'+model_indx+'.root', mcdir+'/Lb2Lcmunu_MagUp_2016_Combine_'+model_indx+'.root']
    tree_num   = 'DecayTree'
    cut_num    = 'isFullsel==1'
    colums_num = ['Event_SM', 'Event_LbProdcorr', 'Event_TrackCalibcorr', 'Event_PIDCalibEffWeight', 'Event_L0Muoncorr', 'Lb_True_Q2_mu' , 'Lb_True_Costhetal_mu', 'isFullsel']
    dfgeom_num = read_root(files_num, key=tree_num, where=cut_num, columns=colums_num) 
    q2_num     = dfgeom_num['Lb_True_Q2_mu'].to_numpy()/1e6
    cthl_num   = dfgeom_num['Lb_True_Costhetal_mu'].to_numpy()
    weights_num= (dfgeom_num['Event_SM']*dfgeom_num['Event_LbProdcorr']*dfgeom_num['Event_TrackCalibcorr']\
                        *dfgeom_num['Event_PIDCalibEffWeight']*dfgeom_num['Event_L0Muoncorr']).to_numpy()
    d_num      = (q2_num, cthl_num, weights_num)
    print('After', dfgeom_num.shape)

    if store_root:
        #make file
        f = TFile.Open("Effs/rootfiles/Eff_Tot_"+model_indx+".root", "recreate")
        f.cd()

        #make histograms
        h_den   = TH2D("hTot"+model_indx+"_den","hTot"+model_indx+"_den;q^{2}[GeV^{2}];cos(#theta_{l});N"          , q2nbins, q2edges, cthlnbins, cthledges)
        h_num   = TH2D("hTot"+model_indx+"_num","hTot"+model_indx+"_num;q^{2}[GeV^{2}];cos(#theta_{l});N(GSRFPTMI)", q2nbins, q2edges, cthlnbins, cthledges)
        h_eff   = TH2D("hTot"+model_indx+"_eff","hTot"+model_indx+"_eff;q^{2}[GeV^{2}];cos(#theta_{l});Eff(Tot)"   , q2nbins, q2edges, cthlnbins, cthledges)

        #fill histograms
        fill_hists(h_den, h_num, h_eff,  d_den, d_num, hdenscale = scale_den)

        h_den.Write()
        h_num.Write()
        h_eff.Write()
    else:
        #fill histograms
        fill_hists(h_den, h_num, h_eff,  d_den, d_num, hdenscale = scale_den)

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
    pickle.dump( Eff, open( './eff_pickled/Eff_'+model_indx+'.p', "wb" ) )
    if store_root:
        f.Close()

    return None

def main():
    GetTotEff()

if __name__ == '__main__':
    model_indx = sys.argv[1]
    store_root = False
    main()
