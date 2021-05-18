import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU
sys.path.append(os.path.abspath(os.getcwd()))
from util import LbToLclNu_Model
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import numpy as np
import matplotlib.pyplot as plt
from root_pandas import to_root, read_root
import pandas as pd
import sys, os
#################################
# Usage: python effmaps_diff_phsp3.py <WC (CVR, CT, CSR, CSL, None)>
################################

def main():
    #input: scenario (CVR, CSR, CSL, CT (No FF varied); None (only FF varied))
    wc = str(sys.argv[1])
    seed = 100

    #import the fit results file
    #with open('plots/results_CVR_' + str(seed) + '_Scheme' + str(scheme) + '_7500000_' + FF + '_toy.txt','r') as txt:
    with open('/home/hep/amathad/Packages/ToyAngular/MC_fitres.txt') as txt:
        data = txt.readlines()
    print(len(data), data) 

    #define dictionary for PDF_SM
    floated_FF = []
    wc_dict_sm = {}
    for i in range(len(data)):
        dataline = data[i].split()
        print(dataline)
        if 'a' in str(dataline[0]):
            floated_FF.append(dataline[0])
        if 'loglh' in str(dataline[0]):
            break
        else:
            wc_dict_sm[str(dataline[0])] = float(dataline[1])
    if wc == 'None':
        wc_dict_sm['WC'] = 0
    else:
        wc_dict_sm[wc] = 0
    
    #add entry to dict_new with cvr value (100 copies of that with different cvr)
    #define the model
    MLb     = 5619.49997776e-3 #GeV
    MLc     = 2286.45992749e-3 #GeV
    Mlep    = 105.6583712e-3   #GeV
    #See util.py for the member variables and member functions for class LbToLclNu_Model e.g. one can obtain
    model   = LbToLclNu_Model(MLb, MLc, Mlep, wc_floated_names = [wc], ff_floated_names = floated_FF)
    if wc == 'None':
        ff_names = model.ff_floated_names
        sample_ff_values = model.sample_ff_values(seed = seed)
    
    #get the parameters to generate phsp array from root file
    fNameU = '/disk/lhcb_data/amathad/Lb2Lclnu_analysis/MC/Lb2Lcmunu_MagUp_2016_Combine.root'
    KeyU     = 'DecayTree'
    ColumnsU    = ['Lb_True_Costhetal_mu', 'Lb_True_Q2_mu', 'runNumber', 'eventNumber']

    fNameD = '/disk/lhcb_data/amathad/Lb2Lclnu_analysis/MC/Lb2Lcmunu_MagDown_2016_Combine.root'
    KeyD      = 'DecayTree'
    ColumnsD    = ['Lb_True_Costhetal_mu', 'Lb_True_Q2_mu', 'runNumber', 'eventNumber']

    fNameSel = '/home/hep/amathad/LbToLclnu_RunTwo/Selection/PID/FFs/LcMuNu_gen_new.root'
    KeySel      = 'MCDecayTree'
    ColumnsSel    = ['Lb_True_Costhetal_mu', 'Lb_True_Q2_mu']

    #get phsp array using the model.import_unbinned_data function (using pathrootfile as pathname input)
    phsp_arr = model.import_unbinned_data(fname = '/home/uzh/hekueh/ToyAngular/plots/toyrootfiles/toysample_'+ str(seed) +'_7500000_toy.root') #add columns here after git pull 
    #phsp_arr = model.import_unbinned_data(fname = fNameU, columns=ColumnsU)
    
    #get PDF_SM
    pdf_sm = (model.get_normalised_pdf_values(phsp_arr, wc_dict_sm)).numpy() #.numpy() function converts tensor array to numpy array, no need for session
    print('Standard Model (SM)', pdf_sm)
    print('Phsp array2:', phsp_arr[0][:].shape, phsp_arr[0][:]) #should be 7,5M rows and 2 col
    
    #get the pdf values for various values of CVR and take difference with respect to SM value    #add entry to dict_new with cvr value (100 copies of that with different cvr)
    nwc = 1
    wcnewvals = np.random.uniform(-0.020, 0.030, nwc) #uniform distribution
    
    #define dict for PDF_NP (WC not zero) and weights
    wc_dict_np = {**wc_dict_sm}
    dt = {}
    for i, wcnewval in enumerate(wcnewvals):
        if wc == 'None':
            wc_dict_np = dict(zip(ff_names, sample_ff_values))
            wc_dict_np['WC'] = 0
        else:
            wc_dict_np[wc] = wcnewval
       
        pdf_np  = (model.get_normalised_pdf_values(phsp_arr, wc_dict_np)).numpy()
        pdf_np = np.array(pdf_np)
        print('New Physics (NP)', pdf_np)
        print('Phsp array:', phsp_arr[0][:].shape, phsp_arr[0][:]) #should be 7,5M rows and 2 col

        #define the weights as the ratio between PDF_NP/PDF_SM
        ratio_val = (pdf_np/pdf_sm)
        print('PDF_NP/PDF_SM:', ratio_val)
        #add weights to the dict
        dt['phsp_weights_' + str(i)] = ratio_val
        
    #create dictionary
    dt['q2']   = phsp_arr[:,0]
    dt['cthmu'] = phsp_arr[:,1]

    #convert dictionary to pandas
    df  = pd.DataFrame.from_dict(dt)
    print('Dataframe:',df)
    print('SM dict:',wc_dict_sm)
    print('NP dict:',wc_dict_np)

    #dump the file to root
    #to_root(df, fname, key=Key, store_index=False)
    #to_root(df, fname, key=Key, store_index=False)
    #to_root(df, fname, key=Key, store_index=False)

    #test root file
    #to_root(df, '/plots/toyrootfiles/test.root', key='DecayTree', store_index=False)

    #import the file from root and print
    #df = read_root('./test.root', columns=['q2', 'cthmu', 'phsp_weights'])
    #df = read_root(fname, columns=Columns)
    #print(df)
    
    exit(1)

    #plot the results
    #define binning scheme
    qsq_min   = Mlep**2
    qsq_max   = (MLb - MLc)**2
    costh_min = -1.
    costh_max =  1.
    qsq_edges = np.array([qsq_min, 0.8, 2.4, 4.8, 7.8, 8.9, 9.8, qsq_max])
    #qsq_inter = mid(qsq_edges)
    costh_edges = np.array([costh_min, -0.75, -0.5, 0, 0.5, costh_max])
    #costh_inter = mid(costh_edges)
    #n_inter1 = len(qsq_inter)
    #n_inter2 = len(costh_inter)
    
    #phsp_arr_mesh = np.array(np.meshgrid(qsq_inter,costh_inter))
    #qsq = phsp_arr_mesh[0]
    #costh = phsp_arr_mesh[1]
    #qsq = qsq.reshape((n_inter1)*(n_inter2),1)
    #costh = costh.reshape((n_inter1)*(n_inter2),1)
    
    phsp_arr = phsp_arr.T
    qsqphsp = phsp_arr[:][0]
    cosphsp = phsp_arr[:][1]
    print(np.shape(phsp_arr[:][0]))

    print('List and weights:',phsp_arr[0][:].shape, phsp_arr[1][:].shape, dt['phsp_weights_0']) 
    n_inter1 = 40
    n_inter2 = 40

    fig,ax = plt.subplots()
    #bin_edges3 = np.array([[qsq_min, 1.2, 3.2, 4.8, 7.8, 8.9, 9.8, qsq_max],[costh_min, -0.9, -0.7, -0.5, -0.1, 0.4, costh_max]])
    bin_edges3 = [qsq_edges, costh_edges]
    ax = plt.hist2d(phsp_arr[0,:].flatten(), phsp_arr[1,:].flatten(), bins = [n_inter1,n_inter2], weights = dt['phsp_weights_0'])
    ax = plt.colorbar()
    ax = plt.title('Maximum Deviation Max(PDF_NP/PDF_SM) of '+str(wc)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
    ax = plt.xlabel('q^2 [GeV^2/c^2]')
    ax = plt.ylabel('cos(theta)')

    fig1, ax = plt.subplots()
    ax = plt.hist2d(phsp_arr[0][:].flatten(), dt['phsp_weights_0'], bins = [40,40])
    ax = plt.colorbar()
    ax = plt.title('Maximum Deviation Max(PDF_NP/PDF_SM) of '+str(wc)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
    ax = plt.xlabel('q^2 [GeV^2/c^2]')
    ax = plt.ylabel('PDF_NP/PDF_SM')

    fig2, ax = plt.subplots()
    ax = plt.hist2d(phsp_arr[1][:].flatten(), dt['phsp_weights_0'], bins = [40, 40])
    ax = plt.colorbar()
    ax = plt.title('Maximum Deviation Max(PDF_NP/PDF_SM) of '+str(wc)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
    ax = plt.xlabel('cos(theta)')
    ax = plt.ylabel('PDF_NP/PDF_SM')

    plt.show()

if __name__ == '__main__':
    main()
