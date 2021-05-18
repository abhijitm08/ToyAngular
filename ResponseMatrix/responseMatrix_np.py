#!/bin/python

from root_pandas import read_root
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import sys
sys.path.append(os.path.abspath(os.getcwd())+'/../')
from BinningSchemes.Binning_Scheme import defing_binning_scheme

bin_scheme= 'Scheme5'
BinScheme = defing_binning_scheme()
q2edges   = BinScheme[bin_scheme]['qsq']
cthledges = BinScheme[bin_scheme]['cthl']
q2nbins   = len(BinScheme[bin_scheme]['qsq'])  - 1
cthlnbins = len(BinScheme[bin_scheme]['cthl']) - 1

#['isTruth', 'isFiducial', 'Lb_True_Q2_mu', 'Lb_True_Costhetal_mu', 'q2_Pred', 'costhl_Pred']

#define bin number
bins_str = '5'

#define cut, bin properties and limits
cut='isTruth==1&&isFiducial==1&&Lb_True_Q2_mu/1e6>=0.0111640356&&Lb_True_Q2_mu/1e6<=11.109822259600001&&Lb_True_Costhetal_mu>=-1.&&Lb_True_Costhetal_mu<=1.'
bins_reco   = [int(bins_str), int(bins_str)]
bins_true   = [int(bins_str), int(bins_str)]
bins        = bins_reco + bins_true
print(bins)
costhl_lim  = (-1., 1.)
q2_lim      = (0.0111640356, 11.109822259600001)
limits      = (q2_lim, costhl_lim, q2_lim, costhl_lim)
print(limits)

#cthl is defined here as angle bw Lc and l in w* rest frame, whereas for ourmodel it is (pi - thisangle)
lget   = ['noexpand:q2_Pred/1e6', 'costhl_Pred', 'noexpand:Lb_True_Q2_mu/1e6', 'Lb_True_Costhetal_mu']
ldict  = {'q2_Pred/1e6':'q2reco', 'costhl_Pred':'cthlreco', 'Lb_True_Q2_mu/1e6':'q2true', 'Lb_True_Costhetal_mu':'cthltrue'}
mcdir  = '/disk/lhcb_data/amathad/Lb2Lclnu_analysis/MC/'
df     = read_root([mcdir+'Lb2Lcmunu_MagDown_2016_Combine.root', mcdir+'Lb2Lcmunu_MagUp_2016_Combine.root'], key='DecayTree', where=cut, columns=lget) 
df     = df.rename(columns=ldict)
l      = ['q2reco', 'cthlreco', 'q2true', 'cthltrue']
print(df[l].values.shape)

#make the 4D histogram
mijkl_norm, edges = np.histogramdd(df[l].values, bins=bins, range=limits) #ALWAYS PASS LIST OF KEYS WHEN CALLING VALUES
print(mijkl_norm.sum())
mijkl_norm = mijkl_norm/mijkl_norm.sum()
print(mijkl_norm.sum())
print(mijkl_norm.shape)
#get bin centers for plotting
f = lambda a: (a[:-1] + a[1:])/2. 
A, B, C, D = f(edges[0]), f(edges[1]), f(edges[2]), f(edges[3]) 
print('A', A)
print('B', B)
print('C', C)
print('D', D)

nkl = np.einsum('ijkl->kl', mijkl_norm) #should be already normalised, sum over ij (reco) and kl (true)
print(nkl.sum())
mijkl_new = mijkl_norm/nkl #convert to probability that a given true value in bin lie in each of the reco bins
nkl_new = np.einsum('ijkl->kl', mijkl_new); print(nkl_new) #should all be one

dirstore = '/home/hep/amathad/LbToLclnu_RunTwo/FittingScripts/qsq_cthl_spectra/Differential_density/responsematrix_eff'
import pickle
pickle.dump(mijkl_new, open(dirstore+'/responsematrix.p', 'wb'))

################# - set things up for plot things
#nmc_true, _ = np.histogramdd(df[l[2:]].values, bins=edges[2:], range=(q2_lim, costhl_lim))
#nmc_true = nmc_true/nmc_true.sum()
#print(nmc_true.sum())
#nmc_reco, _ = np.histogramdd(df[l[:2]].values, bins=edges[:2], range=(q2_lim, costhl_lim))
#nmc_reco = nmc_reco/nmc_reco.sum()
#print(nmc_reco.sum())
#    
#l_toys  = ['q2', 'noexpand:-w_ctheta_l']
#df_toys = read_root('/disk/lhcb_data/amathad/forIaros/Lattice_LbToLcmunu.root', key='tree', columns = l_toys)
#l_dict  = {'-w_ctheta_l' : 'ctheta_l'}
#df_toys = df_toys.rename(columns=l_dict)
#l_use   = ['q2', 'ctheta_l']
#ntrue, _ = np.histogramdd(df_toys[l_use].values, bins=edges[2:], range=(q2_lim, costhl_lim))
#ntrue = ntrue/ntrue.sum()
#print(ntrue.sum())
#nreco = np.einsum('ijkl,kl->ij', mijkl_new, ntrue) #should be normalised
#print(nreco.sum())
#    
#zmin_true = min([np.min(nmc_true), np.min(ntrue)])
#zmax_true = max([np.max(nmc_true), np.max(ntrue)])
#zmin_reco = min([np.min(nmc_reco), np.min(nreco)])
#zmax_reco = max([np.max(nmc_reco), np.max(nreco)])
#    
###############
#fig, axs = plt.subplots(2, 2)
#
#ax = axs[0,0]
#c  = ax.imshow(nmc_true.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[2][0], edges[2][-1], edges[3][0], edges[3][-1]], vmin = zmin_true, vmax = zmax_true)
#ax.set_title('MC True')
#ax.set_ylabel(r'$cos(\theta_l)$')
#cbar = fig.colorbar(c, ax=ax)
#cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#
#ax = axs[0,1]
#c  = ax.imshow(nmc_reco.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]], vmin = zmin_reco, vmax = zmax_reco)
#ax.set_title('MC Reco')
#cbar = fig.colorbar(c, ax=ax)
#cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#
#ax = axs[1,0]
#c  = ax.imshow(ntrue.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[2][0], edges[2][-1], edges[3][0], edges[3][-1]], vmin = zmin_true, vmax = zmax_true)
#ax.set_title('SM PDF True'); 
#ax.set_xlabel('$q^2$')
#ax.set_ylabel(r'$cos(\theta_l)$')
#cbar = fig.colorbar(c, ax=ax)
#cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#
#ax = axs[1,1]
#c  = ax.imshow(nreco.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]], vmin = zmin_reco, vmax = zmax_reco)
#ax.set_title('SM PDF Reco')
#ax.set_xlabel('$q^2$')
#cbar = fig.colorbar(c, ax=ax)
#cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#
#plt.subplots_adjust(hspace=0.35)
#fig.savefig('responseplots/ConvResponseMatrix.pdf')
################
#
#################
#fig, ax = plt.subplots()
#c  = ax.imshow(nmc_true.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[2][0], edges[2][-1], edges[3][0], edges[3][-1]], vmin = zmin_true, vmax = zmax_true)
#ax.set_title('MC True')
#ax.set_xlabel('$q^2$')
#ax.set_ylabel(r'$cos(\theta_l)$')
#cbar = fig.colorbar(c, ax=ax)
#cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.subplots_adjust(hspace=0.35)
#fig.savefig('responseplots/ConvResponseMatrixTrue.pdf')
#
#fig, ax = plt.subplots()
#c  = ax.imshow(nmc_reco.T, aspect = 'auto', interpolation='nearest', origin='lower', extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]], vmin = zmin_reco, vmax = zmax_reco)
#ax.set_title('MC Reco')
#ax.set_xlabel('$q^2$')
#cbar = fig.colorbar(c, ax=ax)
#cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.subplots_adjust(hspace=0.35)
#fig.savefig('responseplots/ConvResponseMatrixReco.pdf')
################
#
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
#fig.savefig('responseplots/ResponseMatrixq2.pdf')
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
#fig.savefig('responseplots/ResponseMatrixcthl.pdf')
################
