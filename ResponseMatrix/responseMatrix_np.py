#!/bin/python

import sys, os
from root_pandas import read_root
import numpy as np
import matplotlib.pyplot as plt
basedir = os.path.abspath(os.getcwd())+'/..'
sys.path.append(basedir)
from BinningSchemes.Binning_Scheme import defing_binning_scheme
import pickle

bin_scheme= 'Scheme5'
BinScheme = defing_binning_scheme()
q2edges   = BinScheme[bin_scheme]['qsq']
cthledges = BinScheme[bin_scheme]['cthl']
q2nbins   = len(BinScheme[bin_scheme]['qsq'])  - 1
cthlnbins = len(BinScheme[bin_scheme]['cthl']) - 1

def get_responce_matrix():
    #define cut, bin properties 
    cut         = 'isTruth==1&&isFiducial==1&&Lb_True_Q2_mu/1e6>=0.0111640356&&Lb_True_Q2_mu/1e6<=11.109822259600001&&Lb_True_Costhetal_mu>=-1.&&Lb_True_Costhetal_mu<=1.'
    bins_reco   = [q2edges, cthledges]
    bins_true   = [q2edges, cthledges]
    bins        = bins_reco + bins_true
    print(bins)
    
    #cthl is defined here as angle bw Lc and l in w* rest frame, whereas for ourmodel it is (pi - thisangle)
    lget       = ['noexpand:q2_Pred/1e6', 'costhl_Pred', 'noexpand:Lb_True_Q2_mu/1e6', 'Lb_True_Costhetal_mu', 'Event_LbProdcorr', 'Event_TrackCalibcorr', 'Event_PIDCalibEffWeight', 'Event_FFcorr']
    files_num  = [basedir+'/model_dependency/model_dependency_rootfiles/Lb2Lcmunu_MagDown_2016_Combine_SM_modeldependency.root']
    files_num += [basedir+'/model_dependency/model_dependency_rootfiles/Lb2Lcmunu_MagUp_2016_Combine_SM_modeldependency.root']

    df         = read_root(files_num, key='DecayTree', where=cut, columns=lget) 
    ldict      = {'q2_Pred/1e6':'q2reco', 'costhl_Pred':'cthlreco', 'Lb_True_Q2_mu/1e6':'q2true', 'Lb_True_Costhetal_mu':'cthltrue'}
    df         = df.rename(columns=ldict)
    phsp_vars  = ['q2reco', 'cthlreco', 'q2true', 'cthltrue']
    weights_num= (df['Event_FFcorr']*df['Event_LbProdcorr']*df['Event_TrackCalibcorr']*df['Event_PIDCalibEffWeight']).to_numpy()
    print(df)
    phsp_reco_true = df[phsp_vars].to_numpy()
    print(phsp_reco_true.shape)
    print(weights_num.shape)
    
    #make the 4D histogram
    mijkl_norm, edges = np.histogramdd(phsp_reco_true, bins=bins, weights = weights_num) 
    print(mijkl_norm.sum())
    mijkl_norm = mijkl_norm/mijkl_norm.sum()
    print(mijkl_norm.sum())
    print(mijkl_norm.shape)
    ##get bin centers for plotting
    #f = lambda a: (a[:-1] + a[1:])/2. 
    #A, B, C, D = f(edges[0]), f(edges[1]), f(edges[2]), f(edges[3]) 
    #print('A', A)
    #print('B', B)
    #print('C', C)
    #print('D', D)
    
    nkl = np.einsum('ijkl->kl', mijkl_norm) #should be already normalised, sum over ij (reco) and kl (true)
    print(nkl.sum())
    mijkl_new = mijkl_norm/nkl #convert to probability that a given true value in bin lie in each of the reco bins
    nkl_new = np.einsum('ijkl->kl', mijkl_new); print(nkl_new) #should all be one
    print('Number of bins', np.sum(mijkl_new))
    exit(1)
    
    pickle.dump(mijkl_new, open('./responsematrix_nominal.p', 'wb'))
    
    ################ - set things up for plot things
    nmc_true, _ = np.histogramdd(phsp_reco_true[:,2:], bins=edges[2:], weights = weights_num)
    nmc_true    = nmc_true/nmc_true.sum()
    print(nmc_true.sum())
    nmc_reco, _ = np.histogramdd(phsp_reco_true[:,:2], bins=edges[:2], weights = weights_num)
    nmc_reco = nmc_reco/nmc_reco.sum()
    print(nmc_reco.sum())
        
    l_toys  = ['q2', 'noexpand:-w_ctheta_l']
    df_toys = read_root('/disk/lhcb_data/amathad/forIaros/Lattice_LbToLcmunu.root', key='tree', columns = l_toys)
    l_dict  = {'-w_ctheta_l' : 'ctheta_l'}
    df_toys = df_toys.rename(columns=l_dict)
    l_use   = ['q2', 'ctheta_l']
    ntrue, _ = np.histogramdd(df_toys[l_use].values, bins=edges[2:])
    ntrue = ntrue/ntrue.sum()
    print(ntrue.sum())
    nreco = np.einsum('ijkl,kl->ij', mijkl_new, ntrue) #should be normalised
    print(nreco.sum())
        
    zmin_true = min([np.min(nmc_true), np.min(ntrue)])
    zmax_true = max([np.max(nmc_true), np.max(ntrue)])
    zmin_reco = min([np.min(nmc_reco), np.min(nreco)])
    zmax_reco = max([np.max(nmc_reco), np.max(nreco)])
        
    ##############
    fig, axs = plt.subplots(2, 2)
    

    ax = axs[0,0]
    #c  = ax.imshow(nmc_true.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[2][0], edges[2][-1], edges[3][0], edges[3][-1]], vmin = zmin_true, vmax = zmax_true)
    c  = ax.pcolormesh(np.meshgrid(q2edges, cthledges)[0], np.meshgrid(q2edges, cthledges)[1], nmc_true.T)
    ax.set_title('MC True')
    ax.set_ylabel(r'$cos(\theta_l)$')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax = axs[0,1]
    #c  = ax.imshow(nmc_reco.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]], vmin = zmin_reco, vmax = zmax_reco)
    c  = ax.pcolormesh(np.meshgrid(q2edges, cthledges)[0], np.meshgrid(q2edges, cthledges)[1], nmc_reco.T)
    ax.set_title('MC Reco')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax = axs[1,0]
    #c  = ax.imshow(ntrue.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[2][0], edges[2][-1], edges[3][0], edges[3][-1]], vmin = zmin_true, vmax = zmax_true)
    c  = ax.pcolormesh(np.meshgrid(q2edges, cthledges)[0], np.meshgrid(q2edges, cthledges)[1], ntrue.T)
    ax.set_title('SM PDF True'); 
    ax.set_xlabel('$q^2$')
    ax.set_ylabel(r'$cos(\theta_l)$')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax = axs[1,1]
    #c  = ax.imshow(nreco.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]], vmin = zmin_reco, vmax = zmax_reco)
    c  = ax.pcolormesh(np.meshgrid(q2edges, cthledges)[0], np.meshgrid(q2edges, cthledges)[1], nreco.T)
    ax.set_title('SM PDF Reco')
    ax.set_xlabel('$q^2$')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.subplots_adjust(hspace=0.35)
    fig.savefig('responsematrix_plots/ConvResponseMatrix_SM_nominal.pdf')
    ###############
    
def get_responce_matrix_alternate_model(scenario, model_indx, conservative):
    #define cut, bin properties 
    cut         = 'isTruth==1&&isFiducial==1&&Lb_True_Q2_mu/1e6>=0.0111640356&&Lb_True_Q2_mu/1e6<=11.109822259600001&&Lb_True_Costhetal_mu>=-1.&&Lb_True_Costhetal_mu<=1.'
    bins_reco   = [q2edges, cthledges]
    bins_true   = [q2edges, cthledges]
    bins        = bins_reco + bins_true
    print(bins)
    
    #cthl is defined here as angle bw Lc and l in w* rest frame, whereas for ourmodel it is (pi - thisangle)
    lget       = ['noexpand:q2_Pred/1e6', 'costhl_Pred', 'noexpand:Lb_True_Q2_mu/1e6', 'Lb_True_Costhetal_mu', 'Event_LbProdcorr', 'Event_TrackCalibcorr', 'Event_PIDCalibEffWeight', 'Event_Model_'+model_indx]
    if conservative:
        files_num  = [basedir+'/model_dependency/model_dependency_rootfiles_conservative/Lb2Lcmunu_MagDown_2016_Combine_'+scenario+'_modeldependency.root']
        files_num += [basedir+'/model_dependency/model_dependency_rootfiles_conservative/Lb2Lcmunu_MagUp_2016_Combine_'+scenario+'_modeldependency.root']
    else:
        files_num  = [basedir+'/model_dependency/model_dependency_rootfiles/Lb2Lcmunu_MagDown_2016_Combine_'+scenario+'_modeldependency.root']
        files_num += [basedir+'/model_dependency/model_dependency_rootfiles/Lb2Lcmunu_MagUp_2016_Combine_'+scenario+'_modeldependency.root']
    
    df         = read_root(files_num, key='DecayTree', where=cut, columns=lget) 
    ldict      = {'q2_Pred/1e6':'q2reco', 'costhl_Pred':'cthlreco', 'Lb_True_Q2_mu/1e6':'q2true', 'Lb_True_Costhetal_mu':'cthltrue'}
    df         = df.rename(columns=ldict)
    phsp_vars  = ['q2reco', 'cthlreco', 'q2true', 'cthltrue']
    weights_num= (df['Event_Model_'+model_indx]*df['Event_LbProdcorr']*df['Event_TrackCalibcorr']*df['Event_PIDCalibEffWeight']).to_numpy()
    print(df)
    phsp_reco_true = df[phsp_vars].to_numpy()
    print(phsp_reco_true.shape)
    print(weights_num.shape)
    
    #make the 4D histogram
    mijkl_norm, edges = np.histogramdd(phsp_reco_true, bins=bins, weights = weights_num) 
    print(mijkl_norm.sum())
    mijkl_norm = mijkl_norm/mijkl_norm.sum()
    print(mijkl_norm.sum())
    print(mijkl_norm.shape)
    ##get bin centers for plotting
    #f = lambda a: (a[:-1] + a[1:])/2. 
    #A, B, C, D = f(edges[0]), f(edges[1]), f(edges[2]), f(edges[3]) 
    #print('A', A)
    #print('B', B)
    #print('C', C)
    #print('D', D)
    
    nkl = np.einsum('ijkl->kl', mijkl_norm) #should be already normalised, sum over ij (reco) and kl (true)
    print(nkl.sum())
    mijkl_new = mijkl_norm/nkl #convert to probability that a given true value in bin lie in each of the reco bins
    nkl_new = np.einsum('ijkl->kl', mijkl_new); print(nkl_new) #should all be one
    
    if conservative:
        dirstore = './responsematrix_pickled_conservative'
    else:
        dirstore = './responsematrix_pickled'

    pickle.dump(mijkl_new, open(dirstore+'/responsematrix_'+scenario+'_'+model_indx+'.p', 'wb'))
    
    ################ - set things up for plot things
    nmc_true, _ = np.histogramdd(phsp_reco_true[:,2:], bins=edges[2:], weights = weights_num)
    nmc_true    = nmc_true/nmc_true.sum()
    print(nmc_true.sum())
    nmc_reco, _ = np.histogramdd(phsp_reco_true[:,:2], bins=edges[:2], weights = weights_num)
    nmc_reco = nmc_reco/nmc_reco.sum()
    print(nmc_reco.sum())
        
    l_toys  = ['q2', 'noexpand:-w_ctheta_l']
    df_toys = read_root('/disk/lhcb_data/amathad/forIaros/Lattice_LbToLcmunu.root', key='tree', columns = l_toys)
    l_dict  = {'-w_ctheta_l' : 'ctheta_l'}
    df_toys = df_toys.rename(columns=l_dict)
    l_use   = ['q2', 'ctheta_l']
    ntrue, _ = np.histogramdd(df_toys[l_use].values, bins=edges[2:])
    ntrue = ntrue/ntrue.sum()
    print(ntrue.sum())
    nreco = np.einsum('ijkl,kl->ij', mijkl_new, ntrue) #should be normalised
    print(nreco.sum())
        
    zmin_true = min([np.min(nmc_true), np.min(ntrue)])
    zmax_true = max([np.max(nmc_true), np.max(ntrue)])
    zmin_reco = min([np.min(nmc_reco), np.min(nreco)])
    zmax_reco = max([np.max(nmc_reco), np.max(nreco)])
        
    ##############
    fig, axs = plt.subplots(2, 2)
    
    ax = axs[0,0]
    c  = ax.imshow(nmc_true.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[2][0], edges[2][-1], edges[3][0], edges[3][-1]], vmin = zmin_true, vmax = zmax_true)
    ax.set_title('MC True')
    ax.set_ylabel(r'$cos(\theta_l)$')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax = axs[0,1]
    c  = ax.imshow(nmc_reco.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]], vmin = zmin_reco, vmax = zmax_reco)
    ax.set_title('MC Reco')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax = axs[1,0]
    c  = ax.imshow(ntrue.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[2][0], edges[2][-1], edges[3][0], edges[3][-1]], vmin = zmin_true, vmax = zmax_true)
    ax.set_title('SM PDF True'); 
    ax.set_xlabel('$q^2$')
    ax.set_ylabel(r'$cos(\theta_l)$')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax = axs[1,1]
    c  = ax.imshow(nreco.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]], vmin = zmin_reco, vmax = zmax_reco)
    ax.set_title('SM PDF Reco')
    ax.set_xlabel('$q^2$')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.subplots_adjust(hspace=0.35)
    if conservative:
        fig.savefig('responsematrix_plots/ConvResponseMatrix_'+scenario+'_'+model_indx+'_conservative.pdf')
    else:
        fig.savefig('responsematrix_plots/ConvResponseMatrix_'+scenario+'_'+model_indx+'.pdf')
    ###############
    
    #################
    #fig, ax = plt.subplots()
    #c  = ax.imshow(nmc_true.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[2][0], edges[2][-1], edges[3][0], edges[3][-1]], vmin = zmin_true, vmax = zmax_true)
    #ax.set_title('MC True')
    #ax.set_xlabel('$q^2$')
    #ax.set_ylabel(r'$cos(\theta_l)$')
    #cbar = fig.colorbar(c, ax=ax)
    #cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #plt.subplots_adjust(hspace=0.35)
    #fig.savefig('responsematrix_plots/ConvResponseMatrixTrue.pdf')
    #
    #fig, ax = plt.subplots()
    #c  = ax.imshow(nmc_reco.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]], vmin = zmin_reco, vmax = zmax_reco)
    #ax.set_title('MC Reco')
    #ax.set_xlabel('$q^2$')
    #ax.set_ylabel(r'$cos(\theta_l)$')
    #cbar = fig.colorbar(c, ax=ax)
    #cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #plt.subplots_adjust(hspace=0.35)
    #fig.savefig('responsematrix_plots/ConvResponseMatrixReco.pdf')
    ################
    
    ################
    #fig, ax = plt.subplots()
    #q = np.einsum('ijkl->ik', mijkl_new)
    #c = ax.imshow(q.T, aspect = 'auto', interpolation=None, origin='lower', extent=[edges[0][0], edges[0][-1], edges[2][0], edges[2][-1]], vmin=np.min(q), vmax = np.max(q))
    #ax.set_title(r'Reponse Matrix')
    #ax.set_xlabel(r'$q^2_{reco}$')
    #ax.set_ylabel(r'$q^2_{true}$')
    #cbar = fig.colorbar(c, ax=ax)
    #cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #plt.subplots_adjust(hspace=0.5)
    #fig.savefig('responsematrix_plots/ResponseMatrixq2.pdf')
    ################
    #
    ################
    #fig, ax = plt.subplots()
    #t = np.einsum('ijkl->jl', mijkl_new)
    #c = ax.imshow(t.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[1][0], edges[1][-1], edges[3][0], edges[3][-1]], vmin=np.min(q), vmax = np.max(q))
    #ax.set_xlabel(r'$cos(\theta_l)_{reco}$')
    #ax.set_ylabel(r'$cos(\theta_l)_{true}$')
    #cbar = fig.colorbar(c, ax=ax)
    #cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #plt.subplots_adjust(hspace=0.5)
    #fig.savefig('responsematrix_plots/ResponseMatrixcthl.pdf')
    ################

def main():
    get_responce_matrix()
    #get_responce_matrix_alternate_model(scenario, model_indx, conservative)

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

    store_root= False
    scenarios = ['CVR', 'CSR', 'CSL', 'CT', 'SM']
    if scenario not in scenarios:
        raise Exception('Scenario not in Scenarios')

    main()
