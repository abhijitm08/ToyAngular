###################
#Inport tensorflow (TF) related things
import os, sys
home = os.getenv('HOME')
sys.path.append(home+"/Packages/TensorFlowAnalysisNew")
import tensorflow as tf
config  = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
from TensorFlowAnalysis import *
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU
#Import global vars and fit models from util.py
sys.path.append(os.path.abspath(os.getcwd()))
from util import * 

import numpy as np
import matplotlib.pyplot as plt
import sys
###################
#Usage: python plot_diff_phsp.py <WC (CVR,CT,CSR,CSL,CVL)> <scheme (1,2,3,4,5,6)>
###################

def mid(a): #calculates mid points in given array
    b = []
    for i in range(len(a)-1):
        d = (a[i+1]-a[i])/2
        m = a[i] + d
        b.append(m)
    return np.array(b)

def main():
    #input is stored in variable
    wcname = sys.argv[1]
    scheme = sys.argv[2]

    #set datatype for arrays
    dt = np.dtype(float)

    #set seed for random numbers
    np.random.seed(100)
    tf.random.set_random_seed(100)

    #define phase space (phsp) limits
    qsq_min   = Mlep**2
    qsq_max   = (MLb - MLc)**2
    costh_min = -1.
    costh_max =  1.
    limts     = [(qsq_min, qsq_max), (costh_min, costh_max)] 
    
    #defint the tensorflow graph of normalised pdf
    pdf_tfgrph =  PDF_TFGraph(limts)
    
    #TODO: 
    #Make the phase space array or the meshgrid of q^2 and costh_mu
    #Here as an example, I have made three data points so the shape of the array is (row x columns) = (3, 2)
    #Note the first columns MUST be qsq in Gev^2 and second column MUST be cos(theta_mu)
    #phsp_arr = np.array([[0.1, 0.2], 
    #                     [1.3, 0.1],
    #                     [5.3,-0.1]])
    #print(phsp_arr.shape)
    
    if scheme == '1':
        n_inter1 = 6
        n_inter2 = n_inter1
        qsq_edges = np.linspace(qsq_min,qsq_max,n_inter1)
        qsq_inter = mid(qsq_edges) 
        costh_edges = np.linspace(costh_min,costh_max,n_inter2)
        costh_inter = mid(costh_edges)
        n_inter1 = len(qsq_inter)
        n_inter2 = len(costh_inter)
    
    elif scheme == '2':
        n_inter1 = 7 #8
        n_inter2 = 3 #6
        qsq_edges = np.linspace(qsq_min,qsq_max,n_inter1)
        qsq_inter = mid(qsq_edges) 
        costh_edges = np.linspace(costh_min,costh_max,n_inter2)
        costh_inter = mid(costh_edges)
        n_inter1 = len(qsq_inter)
        n_inter2 = len(costh_inter)
    
    elif scheme == '3': 
        qsq_edges = np.array([qsq_min, 0.8, 2.4, 4.8, 7.8, 8.9, 9.8, qsq_max])
        qsq_inter = mid(qsq_edges) 
        costh_edges = np.array([costh_min, -0.9, -0.7, -0.5, -0.1, 0.4, costh_max])
        costh_inter = mid(costh_edges)
        n_inter1 = len(qsq_inter)
        n_inter2 = len(costh_inter)
            #bin_edges3 = np.array([[qsq_min, 1.2, 3.2, 4.8, 7.8, 8.9, 9.8, qsq_max],[costh_min, -0.9, -0.7, -0.5, -0.1, 0.4, costh_max]])

    elif scheme == '4':
        qsq_edges = np.array([qsq_min, 0.5, 2, 4.5, 7.5, 9, 10, qsq_max])
        qsq_inter = mid(qsq_edges) 
        costh_edges = np.array([costh_min, -0.75, -0.5, 0, 0.5, costh_max])
        costh_inter = mid(costh_edges)
        n_inter1 = len(qsq_inter)
        n_inter2 = len(costh_inter)
            #bin_edges4 = np.array([[qsq_min, 2, 4, 6, 8, 10, qsq_max],[costh_min, -0.4, -0.1, 0.1, 0.3, 0.7, costh_max]])

    elif scheme == '5': 
        qsq_edges = np.array([qsq_min,0.5, 1.8, 5, 8.5, 9.7, qsq_max])
        qsq_inter = mid(qsq_edges) 
        costh_edges = np.array([costh_min, -0.4, 0, 0.4, costh_max])
        costh_inter = mid(costh_edges)
        n_inter1 = len(qsq_inter)
        n_inter2 = len(costh_inter)
            #bin_edges5 = np.array([[qsq_min, 2, 4, 6, 8, 10, qsq_max],[costh_min, -0.5, 0, 0.5, costh_max]])

    else :
        n_inter1 = 61 #41 - number of data points and bins
        n_inter2 = 61 #41
        qsq_edges = np.linspace(qsq_min,qsq_max,n_inter1)
        qsq_inter = mid(qsq_edges) 
        costh_edges = np.linspace(costh_min,costh_max,n_inter2)
        costh_inter = mid(costh_edges)
        n_inter1 = len(qsq_inter)
        n_inter2 = len(costh_inter)
    
    phsp_arr_mesh = np.array(np.meshgrid(qsq_inter,costh_inter))
    qsq = phsp_arr_mesh[0]
    costh = phsp_arr_mesh[1]
    qsq = qsq.reshape((n_inter1)*(n_inter2),1)
    costh = costh.reshape((n_inter1)*(n_inter2),1)
    phsp_arr = np.concatenate([qsq,costh], axis = 1)
    
    #define a session
    with tf.Session(config=config) as sess:
        #set initial value for the variables. SM values are set for Wilson coefficents and form factors
        sess.run(tf.global_variables_initializer())
    
        #get the SM pdf values at the phase space array setting CVR to zero
        #wcname    = "CVR"
        wcnewval  = 0.00  #set CVR to zero
        pdf_sm    = pdf_tfgrph.values(wcname, wcnewval, phsp_arr, sess)
        print('Standard Model (SM)',pdf_sm)

        #get the pdf values for various values of CVR and take difference with respect to SM value
        #wcnewvals  = [0.01,-0.02]  #TODO: Replace this with sampling of CVR from gaussian of mean of 0.0 and width of 0.03. Here I have made a list of two random values as example.
        nwc = 100
        if wcname == 'CVR':
            wcnewvals = np.random.uniform(-0.020, 0.030, nwc)
        elif wcname == 'CVL':
            #wcnewvals = np.random.uniform(-0.044, 0.020, nwc)
            wcnewvals = np.random.uniform(-0.020, 0.030, nwc)
        elif wcname == 'CSR':
            #wcnewvals = np.random.uniform(-0.460, 0.306, nwc)
            wcnewvals = np.random.uniform(-0.020, 0.030, nwc)
        elif wcname == 'CSL':
            #wcnewvals = np.random.uniform(-0.490, 0.350, nwc)
            wcnewvals = np.random.uniform(-0.020, 0.030, nwc)
        elif wcname == 'CT':
            #wcnewvals = np.random.uniform(-0.050, 0.050, nwc)
            wcnewvals = np.random.uniform(-0.020, 0.030, nwc)
        elif wcname == 'CVRn':
            wcnewvals = np.random.normal(loc = 0.0, scale = 0.03, size = nwc)
            wcname = 'CVR'

        diff_vals  = []
        rel_vals = []
        for wcnewval in wcnewvals:
            #get the new values of pdf when CVR is nonzero
            pdf_np   = pdf_tfgrph.values(wcname, wcnewval, phsp_arr, sess)
            print('New Physics (NP)', pdf_np)
            #Take the absolute value of the (SM - NP) difference
            diff_val = np.abs(pdf_sm - pdf_np)
            rel_val = np.abs(pdf_sm - pdf_np)/pdf_sm
            print('Abs(SM - NP)', diff_val)
            print('Rel(SM - NP)', rel_val)
            #put these values in a list which can later me used to get the maximum of these differences at a given q^2 and cos(theta_mu) point
            diff_vals += [diff_val]
            rel_vals += [rel_val]

        #print(len(diff_vals)) #Since we have used two values of CVR this should be 2.
        print('Difference to SM:')
        print(diff_vals)
        print('Relative difference to SM:')
        print(rel_vals)
        print('PDF SM:')
        print(pdf_sm)

        #TODO: 
        #Get the maximum at deviation at a given q^2 and costhmu point
        #Hint: You can turn the list 'diff_vals' value into a numpy array of shape (row x columns) = (data_size x cvr_values) 
        #      where data_size is the rows of the phsp_arr and cvr_values is the number of CVR values that you have generated. 
        #      Then you should be able to use a numpy 'max' function to get the maximul deviation (See https://numpy.org/doc/stable/reference/generated/numpy.ndarray.max.html)
        diff_vals = np.array(diff_vals).T
        rel_vals = np.array(rel_vals).T
        max_vals = np.max(diff_vals, axis = 1)
        max_rel_vals = np.max(rel_vals, axis = 1)
       # max_vals = max_vals.reshape(phsp_arr_mesh[1].shape)
        
        #print(max_vals, phsp_arr[0], phsp_arr[1])
        #print(max_vals.shape, phsp_arr[0].shape, phsp_arr[1].shape)
        print('WC: ', wcnewvals)
        print(pdf_sm, pdf_np)
        print('SM PDF', pdf_sm[0],pdf_sm[1],pdf_sm[2],pdf_sm[3])
        print('NP PDF', pdf_np[0],pdf_np[1],pdf_np[2],pdf_np[3])
        print('Phasespace (qsq, costh):', qsq[0],costh[0],';',qsq[1],costh[1],';',qsq[2],costh[2],';',qsq[3],costh[3])
        print('Maximum Deviation Overall:', np.max(max_vals))
        
        #TODO
        #Plot the maximum deviation as function of q^2 and cos(theta_mu) using matplotlib
        
        if scheme == '1':
            fig, ax = plt.subplots()
            #equalbin = 5
            #bin_edges1 = np.array([np.linspace(qsq_min,qsq_max,equalbin),np.linspace(costh_min,costh_max,equalbin)])
            bin_edges1 = [qsq_edges, costh_edges]
            ax = plt.hist2d(qsq.flatten(), costh.flatten(), bins = [bin_edges1[0],bin_edges1[1]], weights = max_vals)
            ax = plt.colorbar()
            ax = plt.title('Maximum Deviation of '+str(wcname)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
            ax = plt.xlabel('q^2 [GeV^2/c^2]')
            ax = plt.ylabel('cos(theta)')
            #plt.show()
            
            fig1, ax = plt.subplots()
            ax = plt.hist2d(qsq.flatten(), costh.flatten(), bins = [bin_edges1[0],bin_edges1[1]], weights = max_rel_vals)
            ax = plt.colorbar()
            ax = plt.title('Max(|PDF_SM - PDF_NP|)/PDF_SM of '+str(wcname)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
            ax = plt.xlabel('q^2 [GeV^2/c^2]')
            ax = plt.ylabel('cos(theta)')
            plt.show()

            print(bin_edges1)
        
        elif scheme == '2':
            fig, ax = plt.subplots()
            #equalbin1 = 6
            #equalbin2 = 8
            #bin_edges2 = np.array([np.linspace(qsq_min, qsq_max,equalbin1),np.linspace(costh_min, costh_max,equalbin2)])
            bin_edges2 = [qsq_edges, costh_edges]
            ax = plt.hist2d(qsq.flatten(), costh.flatten(), bins = [bin_edges2[0],bin_edges2[1]], weights = max_vals)
            ax = plt.colorbar()
            ax = plt.title('Maximum Deviation of '+str(wcname)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
            ax = plt.xlabel('q^2 [GeV^2/c^2]')
            ax = plt.ylabel('cos(theta)')
            #plt.show()

            fig1, ax = plt.subplots()
            ax = plt.hist2d(qsq.flatten(), costh.flatten(), bins = [bin_edges2[0],bin_edges2[1]], weights = max_rel_vals)
            ax = plt.colorbar()
            ax = plt.title('Max(|PDF_SM - PDF_NP|)/PDF_SM of '+str(wcname)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
            ax = plt.xlabel('q^2 [GeV^2/c^2]')
            ax = plt.ylabel('cos(theta)')
            plt.show()

            print(bin_edges2)

        elif scheme == '3':
            fig,ax = plt.subplots()
            #bin_edges3 = np.array([[qsq_min, 1.2, 3.2, 4.8, 7.8, 8.9, 9.8, qsq_max],[costh_min, -0.9, -0.7, -0.5, -0.1, 0.4, costh_max]])
            bin_edges3 = [qsq_edges, costh_edges]
            ax = plt.hist2d(qsq.flatten(), costh.flatten(), bins = [bin_edges3[0],bin_edges3[1]], weights = max_vals)
            ax = plt.colorbar()
            ax = plt.title('Maximum Deviation of '+str(wcname)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
            ax = plt.xlabel('q^2 [GeV^2/c^2]')
            ax = plt.ylabel('cos(theta)')
            #plt.show()
            
            fig1, ax = plt.subplots()
            ax = plt.hist2d(qsq.flatten(), costh.flatten(), bins = [bin_edges3[0],bin_edges3[1]], weights = max_rel_vals)
            ax = plt.colorbar()
            ax = plt.title('Max(|PDF_SM - PDF_NP|)/PDF_SM of '+str(wcname)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
            ax = plt.xlabel('q^2 [GeV^2/c^2]')
            ax = plt.ylabel('cos(theta)')
            plt.show()
            print(bin_edges3)
        
        elif scheme == '4':
            fig,ax = plt.subplots()
            #bin_edges4 = np.array([[qsq_min, 2, 4, 6, 8, 10, qsq_max],[costh_min, -0.4, -0.1, 0.1, 0.3, 0.7, costh_max]])
            bin_edges4 = [qsq_edges, costh_edges]
            ax = plt.hist2d(qsq.flatten(), costh.flatten(), bins = [bin_edges4[0],bin_edges4[1]], weights = max_vals)
            ax = plt.colorbar()
            ax = plt.title('Maximum Deviation of '+str(wcname)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
            ax = plt.xlabel('q^2 [GeV^2/c^2]')
            ax = plt.ylabel('cos(theta)')
            #plt.show()
            
            fig1, ax = plt.subplots()
            ax = plt.hist2d(qsq.flatten(), costh.flatten(), bins = [bin_edges4[0],bin_edges4[1]], weights = max_rel_vals)
            ax = plt.colorbar()
            ax = plt.title('Max(|PDF_SM - PDF_NP|)/PDF_SM of '+str(wcname)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
            ax = plt.xlabel('q^2 [GeV^2/c^2]')
            ax = plt.ylabel('cos(theta)')
            plt.show()
            print(bin_edges4)

        elif scheme == '5':
            fig,ax = plt.subplots()
            #bin_edges5 = np.array([[qsq_min, 2, 4, 6, 8, 10, qsq_max],[costh_min, -0.5, 0, 0.5, costh_max]])
            bin_edges5 = [qsq_edges, costh_edges]
            ax = plt.hist2d(qsq.flatten(), costh.flatten(), bins = [bin_edges5[0],bin_edges5[1]], weights = max_vals)
            ax = plt.colorbar()
            ax = plt.title('Maximum Deviation of '+str(wcname)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
            ax = plt.xlabel('q^2 [GeV^2/c^2]')
            ax = plt.ylabel('cos(theta)')
            #plt.show()
            
            fig1, ax = plt.subplots()
            ax = plt.hist2d(qsq.flatten(), costh.flatten(), bins = [bin_edges5[0],bin_edges5[1]], weights = max_rel_vals)
            ax = plt.colorbar()
            ax = plt.title('Max(|PDF_SM - PDF_NP|)/PDF_SM of '+str(wcname)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
            ax = plt.xlabel('q^2 [GeV^2/c^2]')
            ax = plt.ylabel('cos(theta)')
            plt.show()
            print(bin_edges5)

        else:
            fig, ax = plt.subplots()
            #ax = plt.hist2d(qsq.flatten(), cost.flatten(), bins = [n_inter1,n_inter2], weights = max_vals)
            bin_edges = [qsq_edges, costh_edges]
            ax = plt.hist2d(qsq.flatten(), costh.flatten(), bins = [bin_edges[0],bin_edges[1]], weights = max_vals)
            ax = plt.colorbar()
            ax = plt.title('Max(|PDF_SM - PDF_NP|) of '+str(wcname)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
            ax = plt.xlabel('q^2 [GeV^2/c^2]')
            ax = plt.ylabel('cos(theta)')
            #plt.show()
            
            fig, ax = plt.subplots()
            ax = plt.hist2d(qsq.flatten(), costh.flatten(), bins = [bin_edges[0],bin_edges[1]], weights = max_rel_vals)
            ax = plt.colorbar()
            ax = plt.title('Max(|PDF_SM - PDF_NP|)/PDF_SM of '+str(wcname)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
            ax = plt.xlabel('q^2 [GeV^2/c^2]')
            ax = plt.ylabel('cos(theta)')
            #plt.show()
            
            fig, ax = plt.subplots()
            ax = plt.hist2d(qsq.flatten(), costh.flatten(), bins = [bin_edges[0],bin_edges[1]], weights = pdf_sm)
            ax = plt.colorbar()
            ax = plt.title('PDF_SM of '+str(wcname)+' ['+str(n_inter1)+'x'+str(n_inter2)+']')
            ax = plt.xlabel('q^2 [GeV^2/c^2]')
            ax = plt.ylabel('cos(theta)')
            plt.show()
            print(bin_edges)
            #print('Difference at point 3,3:',diff_vals[4][4], ' Difference at 2,2:', diff_vals[1][1], ' Difference at 1,1:', diff_vals[0][0])
            #fig.savefig("plot_diff_phsp.pdf")

if __name__ == '__main__':
    main()
