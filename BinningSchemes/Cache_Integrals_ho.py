import sys, os
import argparse
import tensorflow as tf
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
sys.path.append(os.path.abspath(os.getcwd()))
from Binning_Scheme import defing_binning_scheme
sys.path.append('../')
from util_ho import LbToLclNu_Model
import pickle

def main():
    #set seed and make bin name
    seed    = binnum+10
    tf.random.set_seed(seed)
    np.random.seed(seed)

    #bin name
    bin_name = 'Bin'+str(binnum)
    bin_lmts = BinScheme[scheme]['bin_limits'][bin_name]
    
    #Define the Model with what needs to be floated (can actually be done later too if not done in the declaration)
    MLb     = 5619.49997776e-3 #GeV
    MLc     = 2286.45992749e-3 #GeV
    Mlep    = 105.6583712e-3   #GeV
    model   = LbToLclNu_Model(MLb, MLc, Mlep, get_HO_FF = True) 

    ##############
    #generate normal sample in this range
    model.phase_space.ranges = bin_lmts
    norm_smpl  = model.phase_space.unfiltered_sample(1000000)
    fpdinp     = model.get_freeparams_indp_terms(norm_smpl, getIntegral = True) 
    ks, vs     = list(fpdinp.keys()), list(fpdinp.values())
    print(len(ks))
    savedir = os.path.abspath(os.getcwd()+'/'+scheme)
    if bin_name == 'Bin0': pickle.dump( ks, open(savedir+'_ho/keys'+str(binnum)+'.p', "wb" ))
    pickle.dump( vs, open(savedir+'_ho/Bin'+str(binnum)+'.p', "wb" ))
    #############
    
    ################ Test 1 - Actual density values
    #phsp_arr = np.array([[0.1, 0.2], 
    #                     [1.3, 0.1],
    #                     [5.3,-0.1]])

    #pdfa = model.get_unbinned_model(phsp_arr, method= '1')
    #print(pdfa)

    #pdfb = model.get_unbinned_model(phsp_arr, method= '2')
    #print(pdfb)
    ################
    
    ################# Test 2 - Actual integral values inside a given bin
    #model.phase_space.ranges = bin_lmts

    #norm_smpl  = model.phase_space.unfiltered_sample(1000000)
    #intga = tf.reduce_mean(model.get_unbinned_model(norm_smpl, method= '1'))
    #print(intga)

    #intgb = tf.reduce_mean(model.get_unbinned_model(norm_smpl, method= '2'))
    #print(intgb)
    #################
    
    ################# Test 2 - Import the cached integral and check against unbinned
    #model.phase_space.ranges = bin_lmts
    #norm_smpl  = model.phase_space.unfiltered_sample(1000000)
    #intga      = tf.reduce_mean(model.get_unbinned_model(norm_smpl, method= '1'))
    #print(intga)

    #savedir   = os.path.abspath(os.getcwd()+'/'+scheme)
    #np_fpdinps= pickle.load(open(savedir+'/Bin'+str(binnum)+'.p','rb'))
    #k_fpdinp  = pickle.load(open(savedir+'/keys0.p', 'rb'))
    #print(len(np_fpdinps))
    #print(len(k_fpdinp))
    #fpdinp    = dict(zip(k_fpdinp,np_fpdinps))
    #intgb     = model.contract_with_freeparams(fpdinp) #contract with the wcs
    #print(intgb)
    ##################
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for Cache_Integrals.py')
    parser.add_argument('-b', '--binnum', dest='binnum',type=int,required=True, help='(string) Either a list or just one bin number from the scheme. User needs to know appriori how many bins are there in the scheme (See Binning_Schemes.py).')
    parser.add_argument('-s', '--scheme', dest='scheme',type=str, required=True, help='(string) The binning scheme to use. This scheme needs to have been implemented in Binning_Schemes.py file. Available options are Scheme{0,1,2,3,4,5,6}.')
    args       = parser.parse_args()
    scheme     = args.scheme
    binnum     = args.binnum
    BinScheme  = defing_binning_scheme()
    if scheme not in list(BinScheme.keys()): 
        raise Exception('The specified binning scheme does not exist!')

    total_bins = (len(BinScheme[scheme]['qsq']) - 1) * (len(BinScheme[scheme]['cthl']) - 1)
    if binnum >= total_bins:
        raise Exception('The specified bin number does not exist')

    print(args)
    main()
