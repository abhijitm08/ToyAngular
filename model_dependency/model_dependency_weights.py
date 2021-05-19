import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU
sys.path.append(os.path.abspath(os.getcwd())+'/../')
from util import LbToLclNu_Model
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import numpy as np
import matplotlib.pyplot as plt
from root_pandas import to_root, read_root
import pandas as pd

def fill_weights(scenario, df_phsp_arr, dict_params_pdf_old, n_params = 100):
    #get the phase space variables
    phsp_arr_q2   = df_phsp_arr['Lb_True_Q2_mu'].to_numpy() * 1e-6 #convert to GeV^2
    phsp_arr_cthl = df_phsp_arr['Lb_True_Costhetal_mu'].to_numpy()
    phsp_arr      = np.concatenate([phsp_arr_q2.reshape(-1,1), phsp_arr_cthl.reshape(-1,1)], axis=1)
    print(phsp_arr)
    
    #define the model
    MLb     = 5619.49997776e-3 #GeV
    MLc     = 2286.45992749e-3 #GeV
    Mlep    = 105.6583712e-3   #GeV
    #See util.py for the member variables and member functions for class LbToLclNu_Model e.g. one can obtain
    wc_name = scenario
    if scenario == 'SM': wc_name = 'None'
    model   = LbToLclNu_Model(MLb, MLc, Mlep, wc_floated_names = [wc_name], ff_floated_names = ['All'])
    
    #get PDF_SM
    pdf_old = (model.get_normalised_pdf_values(phsp_arr, dict_params_pdf_old)).numpy() #.numpy() function converts tensor array to numpy array, no need for session
    #print('Old pdf vals', pdf_old)

    #define weights for SM i.e. FF to their central values
    pdf_sm = (model.get_normalised_pdf_values(phsp_arr, {})).numpy()
    #print('SM pdf vals', pdf_old)
    df_phsp_arr['Event_FFcorr'] = pdf_sm/pdf_old
    #print('weights_sm:', df_phsp_arr['Event_FFcorr'])

    #make different values of params according to scenario
    if scenario == 'CVR':
        new_param_vals = np.random.uniform(-0.020, 0.030, n_params)
    elif scenario == 'CSR':
        new_param_vals = np.random.uniform(-0.460, 0.306, n_params)
    elif scenario == 'CSL':
        new_param_vals = np.random.uniform(-0.490, 0.350, n_params)
    elif scenario == 'CT':
        new_param_vals = np.random.uniform(-0.050, 0.050, n_params)
    elif scenario == 'SM':
        new_param_vals = model.sample_ff_values(seed = seed, size = n_params, verbose = False)

    #print('New samples of params', new_param_vals)
    
    #define dict for PDF_NP (WC not zero) and weights
    for i, new_param_val in enumerate(new_param_vals):
        dict_params_pdf_new = None
        if scenario == 'SM':
            dict_params_pdf_new = dict(zip(model.ff_floated_names, new_param_val))
        else:
            dict_params_pdf_new = {wc_name : new_param_val}
       
        pdf_new = (model.get_normalised_pdf_values(phsp_arr, dict_params_pdf_new)).numpy()
        pdf_new = np.array(pdf_new)
        #print('New PDF vals', pdf_new)

        #define the weights as the ratio between PDF_NP/PDF_SM
        ratio_val = (pdf_new/pdf_old)
        #print('PDF_NP/PDF_SM:', ratio_val)
        #add weights to the dict
        df_phsp_arr['Event_Model_'+str(i)] = ratio_val
        print(i)
        
def main():
    #get the observables from the MC root files
    if file_type == 'Signal_MU':
        fname   = '/disk/lhcb_data/amathad/Lb2Lclnu_analysis/MC/Lb2Lcmunu_MagUp_2016_Combine.root'
        key     = 'DecayTree'
        reco_truth_vars = ['Lb_True_Q2_mu', 'Lb_True_Costhetal_mu', 'q2_Pred', 'costhl_Pred']
        extra_sel_vars  = ['isTruth', 'isFiducial', 'Event_LbProdcorr', 'Event_TrackCalibcorr', 'Event_PIDCalibEffWeight', 'Event_L0Muoncorr', 'isFullsel', 'runNumber', 'eventNumber']
        columns = reco_truth_vars + extra_sel_vars
    elif file_type == 'Signal_MD':
        fname   = '/disk/lhcb_data/amathad/Lb2Lclnu_analysis/MC/Lb2Lcmunu_MagDown_2016_Combine.root'
        key     = 'DecayTree'
        reco_truth_vars = ['Lb_True_Q2_mu', 'Lb_True_Costhetal_mu', 'q2_Pred', 'costhl_Pred']
        extra_sel_vars  = ['isTruth', 'isFiducial', 'Event_LbProdcorr', 'Event_TrackCalibcorr', 'Event_PIDCalibEffWeight', 'Event_L0Muoncorr', 'isFullsel', 'runNumber', 'eventNumber']
        columns = reco_truth_vars + extra_sel_vars
    elif file_type == 'Gen':
        fname   = '/home/hep/amathad/LbToLclnu_RunTwo/FittingScripts/qsq_cthl_spectra/Differential_density/responsematrix_eff/GeomEffFiles/LcMuNu_gen_new.root'
        key     = 'DecayTree'
        columns = ['Lb_True_Costhetal_mu', 'Lb_True_Q2_mu', 'Event_LbProdcorr']

    #get phsp array using the model.import_unbinned_data function (using pathrootfile as pathname input)
    df_phsp_arr   = read_root(fname, key = key, columns = columns)

    #import the fit results file for PDF_OLD and make a dictionary
    with open('./MC_fitres.txt') as txt: data = txt.readlines()
    print(len(data), data) 
    dict_params_pdf_old = {}
    for i in range(len(data)):
        dataline = data[i].split()
        print(dataline)
        if 'loglh' in str(dataline[0]):
            break
        else:
            dict_params_pdf_old[str(dataline[0])] = float(dataline[1])

    print(dict_params_pdf_old)

    #fill with weights
    n_params = 100
    fill_weights(scenario, df_phsp_arr, dict_params_pdf_old, n_params = n_params)
    print(df_phsp_arr)

    #dump the file to root
    f_new_name = './model_dependency_rootfiles_new/'+fname.split('/')[-1]
    f_new_name = f_new_name.replace('.root', '_'+scenario+'_modeldependency.root')
    print(f_new_name)
    to_root(df_phsp_arr, f_new_name, key=key, store_index=False)

##plot the results
##define binning scheme
#qsq_min   = Mlep**2
#qsq_max   = (MLb - MLc)**2
#costh_min = -1.
#costh_max =  1.
#qsq_edges = np.array([qsq_min, 0.8, 2.4, 4.8, 7.8, 8.9, 9.8, qsq_max])
##qsq_inter = mid(qsq_edges)
#costh_edges = np.array([costh_min, -0.75, -0.5, 0, 0.5, costh_max])
##costh_inter = mid(costh_edges)
##n_inter1 = len(qsq_inter)
##n_inter2 = len(costh_inter)
#
##phsp_arr_mesh = np.array(np.meshgrid(qsq_inter,costh_inter))
##qsq = phsp_arr_mesh[0]
##costh = phsp_arr_mesh[1]
##qsq = qsq.reshape((n_inter1)*(n_inter2),1)
##costh = costh.reshape((n_inter1)*(n_inter2),1)
#
#phsp_arr = phsp_arr.T
#qsqphsp = phsp_arr[:][0]
#cosphsp = phsp_arr[:][1]
#print(np.shape(phsp_arr[:][0]))

#print('List and weights:',phsp_arr[0][:].shape, phsp_arr[1][:].shape, dt['phsp_weights_0']) 
#n_inter1 = 40
#n_inter2 = 40

#fig,ax = plt.subplots()
##bin_edges3 = np.array([[qsq_min, 1.2, 3.2, 4.8, 7.8, 8.9, 9.8, qsq_max],[costh_min, -0.9, -0.7, -0.5, -0.1, 0.4, costh_max]])
#bin_edges3 = [qsq_edges, costh_edges]
#ax = plt.hist2d(phsp_arr[0,:].flatten(), phsp_arr[1,:].flatten(), bins = [n_inter1,n_inter2], weights = dt['phsp_weights_0'])
#ax = plt.colorbar()
#ax = plt.title('Maximum Deviation Max(PDF_NP/PDF_SM) of '+str(wc)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
#ax = plt.xlabel('q^2 [GeV^2/c^2]')
#ax = plt.ylabel('cos(theta)')

#fig1, ax = plt.subplots()
#ax = plt.hist2d(phsp_arr[0][:].flatten(), dt['phsp_weights_0'], bins = [40,40])
#ax = plt.colorbar()
#ax = plt.title('Maximum Deviation Max(PDF_NP/PDF_SM) of '+str(wc)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
#ax = plt.xlabel('q^2 [GeV^2/c^2]')
#ax = plt.ylabel('PDF_NP/PDF_SM')

#fig2, ax = plt.subplots()
#ax = plt.hist2d(phsp_arr[1][:].flatten(), dt['phsp_weights_0'], bins = [40, 40])
#ax = plt.colorbar()
#ax = plt.title('Maximum Deviation Max(PDF_NP/PDF_SM) of '+str(wc)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
#ax = plt.xlabel('cos(theta)')
#ax = plt.ylabel('PDF_NP/PDF_SM')

#plt.show()

if __name__ == '__main__':
    file_type  = str(sys.argv[1])
    scenario   = str(sys.argv[2])
    seed       = 100

    scenarios = ['CVR', 'CSR', 'CSL', 'CT', 'SM']
    if scenario not in scenarios:
        raise Exception('Scenario not in Scenarios')

    main()
