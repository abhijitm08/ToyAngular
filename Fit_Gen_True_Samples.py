#!/bin/python
import sys, os
home = os.getenv('HOME')
sys.path.append(home+"/Packages/AmpliTF/")
import amplitf.kinematics as atfk
import root_pandas as rpd
import tensorflow as tf
import numpy as np
from util import LbToLclNu_Model, Minimize

def get_phasespace_vars(PLb_lab, PLc_lab, PLepton_lab):
    PW_lab       = PLb_lab - PLc_lab
    PLb_Wlab     = atfk.boost_to_rest(PLb_lab, PW_lab)                                               
    PLepton_Wlab = atfk.boost_to_rest(PLepton_lab, PW_lab)                            
    q2           = atfk.mass(PW_lab)**2/1e6 #Convert to GeV^2
    costhl       = atfk.scalar_product(atfk.unit_vector(atfk.spatial_components(PLepton_Wlab)), -1. * atfk.unit_vector(atfk.spatial_components(PLb_Wlab))) 
    return q2, costhl

def get_gen_sample(sample= 'mu'):
    den_columns  = []
    den_columns += ['Lambda_b0_TRUEP_E']
    den_columns += ['Lambda_b0_TRUEP_X']
    den_columns += ['Lambda_b0_TRUEP_Y']
    den_columns += ['Lambda_b0_TRUEP_Z']
    den_columns += ['Lambda_cplus_TRUEP_E']
    den_columns += ['Lambda_cplus_TRUEP_X']
    den_columns += ['Lambda_cplus_TRUEP_Y']
    den_columns += ['Lambda_cplus_TRUEP_Z']
    den_columns += [sample+'minus_TRUEP_E']
    den_columns += [sample+'minus_TRUEP_X']
    den_columns += [sample+'minus_TRUEP_Y']
    den_columns += [sample+'minus_TRUEP_Z']
    den_columns += ['nu_'+sample+'~_TRUEP_E']
    den_columns += ['nu_'+sample+'~_TRUEP_X']
    den_columns += ['nu_'+sample+'~_TRUEP_Y']
    den_columns += ['nu_'+sample+'~_TRUEP_Z']
    den_fname    = '~/LbToLclnu_RunTwo/Selection/PID/FFs/GenMC/Lc'+sample.capitalize()+'Nu_gen.root'
    
    df_den     = rpd.read_root(den_fname, columns = den_columns, key='MCDecayTreeTuple/MCDecayTree')

    PLc_lab    = atfk.lorentz_vector(atfk.vector(df_den['Lambda_cplus_TRUEP_X']   , df_den['Lambda_cplus_TRUEP_Y']   , df_den['Lambda_cplus_TRUEP_Z'])   , df_den['Lambda_cplus_TRUEP_E']) 
    Pl_lab     = atfk.lorentz_vector(atfk.vector(df_den[sample+'minus_TRUEP_X']   , df_den[sample+'minus_TRUEP_Y']   , df_den[sample+'minus_TRUEP_Z'])   , df_den[sample+'minus_TRUEP_E'])
    PNu_lab    = atfk.lorentz_vector(atfk.vector(df_den["nu_"+sample+"~_TRUEP_X"] , df_den["nu_"+sample+"~_TRUEP_Y"] , df_den["nu_"+sample+"~_TRUEP_Z"]) , df_den["nu_"+sample+"~_TRUEP_E"])
    PLb_lab    = PLc_lab + Pl_lab + PNu_lab
    df_den['Lb_True_Q2'], df_den['Lb_True_Costhetal'] = get_phasespace_vars(PLb_lab, PLc_lab, Pl_lab)
    return df_den[['Lb_True_Q2', 'Lb_True_Costhetal']].to_numpy(), 

def main():
    #set seed
    seed = 10
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    #generate list of form factors to be floated (note fixed one a0gplus since not sensitive and correlated with others).
    ff_params = ['a0f0', 'a0fplus', 'a0fperp', 'a0g0', 'a1f0', 'a1fplus', 'a1fperp', 'a1g0', 'a1gplus', 'a1gperp']
    
    #Define the Model and unbinned nll
    MLb     = 5619.49997776e-3 #GeV
    MLc     = 2286.45992749e-3 #GeV
    Mlep    = 105.6583712e-3   #GeV
    model   = LbToLclNu_Model(MLb, MLc, Mlep, wc_floated_names = ['None'], ff_floated_names = ff_params) 
    
    #get real data
    gen_sample = (get_gen_sample()[0])[:20000,:]
    print(gen_sample)
    print(gen_sample.shape)
    #gen fake data
    #gen_sample = model.generate_unbinned_data(size = 10000, seed = seed, chunks = 1000000, store_file = False)
    #print(gen_sample)
    
    #define NLL values
    nll     = model.unbinned_nll(gen_sample, method='1')
    tot_params = model.tot_params 
    for k,v in tot_params.items(): print(k, v.numpy())
    print('NLL: ', nll(tot_params).numpy()) 
    #d = {'a0f0': 1.2}
    #model.set_params_values(d, isfitresult = False)
    #for k,v in tot_params.items(): print(k, v.numpy())
    #print('NLL: ', nll(tot_params).numpy()) 
    #exit(1)
    
    reslts = Minimize(nll, model, tot_params, nfits = 1, use_hesse = True, use_minos = False, use_grad = True, randomiseFF = True, get_covariance = False)
    print(reslts)

    #plot
    fitres = model.generate_unbinned_data(size = 10*gen_sample.shape[0], seed = seed, chunks = 1000000, store_file = False)
    model.plot_fitresult_unbinned(gen_sample, fitres, 'plots/fitres.pdf', bin_scheme = 'Scheme7', xlabel = "$q^{2} [GeV^{2}]$", ylabel = r"$cos(\theta_{\mu})$")
    for k,v in tot_params.items(): print(k, v.numpy())
    print('NLL: ', nll(tot_params).numpy()) 

if __name__ == '__main__':
    main()
