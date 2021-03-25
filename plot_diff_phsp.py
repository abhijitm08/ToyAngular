import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU
sys.path.append(os.path.abspath(os.getcwd()))
from util import LbToLclNu_Model
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import numpy as np

def main():
    #Define the model
    MLb     = 5619.49997776e-3 #GeV
    MLc     = 2286.45992749e-3 #GeV
    Mlep    = 105.6583712e-3   #GeV
    model   = LbToLclNu_Model(MLb, MLc, Mlep) 

    #See util.py for the member variables and member functions for class LbToLclNu_Model e.g. one can obtain
    phsp_limts= model.phase_space.ranges
    qsq_min   = phsp_limts[0][0]
    qsq_max   = phsp_limts[0][1]
    costh_min = phsp_limts[1][0]
    costh_max = phsp_limts[1][1]

    #Make the phase space array or the meshgrid of q^2 and costh_mu
    phsp_arr = np.array([[0.1, 0.2], 
                         [1.3, 0.1],
                         [5.3,-0.1]])
    print(phsp_arr.shape)
    
    #get the SM pdf values at the phase space array setting CVR to zero
    wcname    = "CVR"
    wcnewval  = 0.00  #set CVR to zero
    wc_dict   = {wcname : wcnewval} #make a dictionary
    pdf_sm    = (model.get_normalised_pdf_values(phsp_arr, wc_dict)).numpy() #.numpy() function converts tensor array to numpy array, no need for session
    print('Standard Model (SM)', pdf_sm)

    #get the pdf values for various values of CVR and take difference with respect to SM value
    wcnewvals  = [0.01,-0.02]  
    diff_vals  = []
    for wcnewval in wcnewvals:
        #get the new values of pdf when CVR is nonzero
        wc_dict = {wcname : wcnewval} #make a dictionary
        pdf_np  = (model.get_normalised_pdf_values(phsp_arr, wc_dict)).numpy()
        print('New Physics (NP)', pdf_np)
        #Take the absolute value of the (SM - NP) difference
        diff_val = np.abs(pdf_sm - pdf_np) 
        print('Abs(SM - NP)', diff_val)
        #put these values in a list which can later me used to get the maximum of these differences at a given q^2 and cos(theta_mu) point
        diff_vals += [diff_val]

    print(len(diff_vals)) 

if __name__ == '__main__':
    main()
