import sys, os
import time, pprint
import argparse
import tensorflow as tf
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
sys.path.append(os.path.abspath(os.getcwd()))
from util import LbToLclNu_Model, str2bool, Minimize
fitdir=os.path.dirname(os.path.abspath(__file__))
home = os.getenv('HOME')
sys.path.append(home+"/Packages/TFA2/")

def main():
    start = time.time()

    #set seed
    tf.random.set_seed(seed)
    np.random.seed(seed)

    #Define the Model with what needs to be floated (can actually be done later too if not done in the declaration)
    MLb     = 5619.49997776e-3 #GeV
    MLc     = 2286.45992749e-3 #GeV
    Mlep    = 105.6583712e-3   #GeV
    model   = LbToLclNu_Model(MLb, MLc, Mlep, wc_floated_names = [floatWC], ff_floated_names = floated_FF) 

    #Get a binned Model (it actually returns a function that takes no arguments, this is required by tensorflow2)
    b_model = model.get_binned_model(bin_scheme = bscheme, applyEff = effn, applyResponse = resn, eff_fname = effpath, res_fname = respath)
    #print(b_model())
    
    #Generate a binned data
    ##method1: Use "binned" pdf where each bin entry is sampled from a poisson distribution
    #b_data  = model.generate_binned_data(nevnts, b_model, seed = seed) 
    #method2: Use "unbinned" pdf to generate data (using accept-reject method) and then bin it.
    #NB: For investigating binning schemes we will be fitting the same sample with different schemes so generate one sample and store it
    rootfiledir   = direc+'toyrootfiles/'
    sample_fname  = rootfiledir+'toysample_'+str(seed)+'_'+str(nevnts)+'_'+suffix+'.root'
    if not os.path.exists(rootfiledir):
        print('Making directory ', direc+'toyrootfiles')
        os.system('mkdir '+direc+'toyrootfiles')

    if os.path.isfile(sample_fname):
        print('Importing the file', sample_fname)
        b_data  = model.generate_binned_data_alternate(nevnts, bscheme, seed = seed, import_file = True, store_file = False, fname = sample_fname) #
    else:
        print('Making a new file', sample_fname)
        b_data  = model.generate_binned_data_alternate(nevnts, bscheme, seed = seed, import_file = False, store_file = True, fname = sample_fname) #

    #Define the negative log-likilihood with/without the gaussian constrain on the floated form factors. 
    #Actually it returns a function that takes dictionary of parameters as input (a requirement by TFA2 package).
    nll = model.binned_nll(b_data, b_model, gauss_constraint_ff = True)
    tot_params = model.tot_params #get both fixed and floated parameter dictionary
    print('NLL', nll(tot_params)) #print the nll value of the model given the binned data with the parameters with which the model was created

    #For a given data sample (b_data) one would conduct nfits with different starting values for the free parameters and use the results of the fit that gives the least nll value.
    #This will ensure that we are converging to a global minima. The default value of nfits is 1, so for now only one fit is conducted. 
    reslts = Minimize(nll, model, tot_params, nfits = nfits, use_hesse = True, use_minos = False)
    #pprint.pprint(reslts)

    ##write the fit results to a text file
    resname = 'results_'+floatWC+'_'+str(seed)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix
    resfname = direc+resname+'.txt'
    print('resfname', resfname)
    model.write_fit_results(reslts, resfname)

    #plot the fit results
    plotfname = direc+resname+'.pdf'
    b_fit = b_model().numpy() #After the setting the parameter values to the estimated ones (as above), we get here the predicted shape
    if plotRes: model.plot_fitresult(b_data, b_fit, bin_scheme = bscheme, fname = plotfname)
    
    end = time.time(); print('Time taken in min', (end - start)/60.)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for LbToLclnu_fit_bscheme.py')
    #required aguments
    parser.add_argument('-f', '--floatWC'   , dest='floatWC',type=str, required=True, help='(string) Name of the Wilson coefficient (WC) to be floated. Available options are [CVR,CSR,CSL,CT].')
    parser.add_argument('-s', '--seed'      , dest='seed'   ,type=int, required=True, help='(int) Seed for generation of fake/toy data. This should be different for each toy.')
    #optional arguments
    parser.add_argument('-b', '--bscheme'   , dest='bscheme',type=str, default='Scheme0',help='(string) Binning scheme to be used. Available options are [Scheme0,Scheme1,Scheme2,Scheme3,Scheme4,Scheme5,Scheme6] and default is Scheme0.')
    parser.add_argument('-n', '--nevnts'    , dest='nevnts' ,type=int, default=int(7.5e6),help='(int) Size of the toy sample. Default is 7.5M events.')
    parser.add_argument('-nf','--nfits'     , dest='nfits'  ,type=int, default=1,help='(int) Number of fits to conduct to a given sample. Default in 1.')
    parser.add_argument('-sf','--suffix'    , dest='suffix' ,type=str, default='toy',help="(int) A unique suffix added to the name of the fit result file (*_suffix.txt) and plot file (*_suffix.pdf). Default is 'toy'.")
    parser.add_argument('-d', '--direc'     , dest='direc'  ,type=str, default='./plots/',help='(string) Directory in which the fit result (.txt) and plot is to be saved. Default in current directory.')
    parser.add_argument('-p', '--plotRes'   , dest='plotRes',type=str2bool,default='True',help='(bool) Set to False if you do not want to plot the result. Default is True.')
    parser.add_argument('-effn', '--effn'   , dest='effn'   ,type=str2bool,default='False',help='(bool) Set to True if you want efficiency included in model. Default is False.')
    parser.add_argument('-effp', '--effpath', dest='effpath',type=str, default=fitdir+'/responsematrix_eff/Eff.p',help='(string) Path to efficiency file. Default is: '+fitdir+'/responsematrix_eff/Eff.p')
    parser.add_argument('-resn', '--resn'   , dest='resn'   ,type=str2bool,default='False',help='(bool) Set to True if you want resolution information included in model. Default is False.')
    parser.add_argument('-resp', '--respath', dest='respath',type=str,default=fitdir+'/responsematrix_eff/responsematrix.p',help='(bool) Path to resoponse matrix file. Default is:'+fitdir+'/responsematrix_eff/responsematrix.p')
    parser.add_argument('-e', '--floated_FF' , dest='floated_FF',nargs='+', default = ['None'], 
    help="(list) List of form factor (FF) parameters that you want floated in the fit. \
          Default is 'None' that is all FF parameters are fixed. \
          When CVR or CSR or CSL is set as 'floatWC': 11 FF parameters can be floated, they are a0f0 a0fplus a0fperp a1f0 a1fplus a1fperp a0g0 a0gplus a1g0 a1gplus a1gperp \
          When CT is set as 'floatWC': In addition to the 11 FF, we can float 7 more which are a0hplus a0hperp a0htildeplus a1hplus a1hperp a1htildeplus a1htildeperp") 
    args       = parser.parse_args()
    floatWC    = args.floatWC
    seed       = args.seed
    bscheme    = args.bscheme
    nevnts     = args.nevnts
    nfits      = args.nfits
    suffix     = args.suffix
    direc      = args.direc
    plotRes    = args.plotRes
    effn       = args.effn
    effpath    = args.effpath
    resn       = args.resn
    respath    = args.respath
    floated_FF = args.floated_FF
    print(args)
    if not direc.endswith('/'): direc += '/'
    main()
