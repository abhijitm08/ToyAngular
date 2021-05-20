import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU
import time, pprint
import argparse
import numpy as np
import tensorflow as tf
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

    #Get a binned Model one to generate (it actually returns a function that takes no arguments, this is required by tensorflow2)
    b_model_gen = model.get_binned_model(bin_scheme = bscheme, applyEff = True, applyResponse = True, eff_fname = fitdir+'/Eff/Eff_Tot_SM_nominal.p', res_fname = fitdir+'/ResponseMatrix/responsematrix_nominal.p' )
    #print(b_model_gen())

    #set the values of the parameters to the one passed in, before generating toy
    if gen_param_vals is not None:
        model.set_params_values(gen_param_vals, isfitresult = False)
    
    #Generate a binned data
    b_data = None
    if unbinned_toygen:
        #Use "unbinned" pdf to generate data (using accept-reject method) and then bin it.
        #NB: For investigating binning schemes we will be fitting the same sample with different schemes so generate one sample and store it
        rootfiledir   = direc+'toyrootfiles/'
        sample_fname  = rootfiledir+'toysample_'+str(seed)+'_'+str(nevnts)+'_'+suffix+'.root'
        if os.path.isfile(sample_fname):
            binnedfname = sample_fname.replace('.root', '_'+bscheme+'_binned.npy')
            if os.path.isfile(binnedfname):
                print('Importing the binned file', binnedfname)
                b_data  = np.load(binnedfname)
            else:
                print('Importing the root file', sample_fname)
                b_data  = model.generate_binned_data_alternate(nevnts, bscheme, seed = seed, import_file = True, store_file = False, fname = sample_fname) #
                print('Saving binned numpy histogram', binnedfname)
                np.save(binnedfname, b_data)
        else:
            print('Making a new file', sample_fname)
            b_data  = model.generate_binned_data_alternate(nevnts, bscheme, seed = seed, import_file = False, store_file = True, fname = sample_fname)
            print('Saving binned numpy histogram', binnedfname)
            np.save(binnedfname, b_data)
    else:
        #Use "binned" pdf where each bin entry is sampled from a poisson distribution
        b_data  = model.generate_binned_data(nevnts, b_model_gen, seed = seed) 

    #print('Binned data is', b_data)

    #Define the negative log-likilihood with/without the gaussian constrain on the floated form factors. 
    #Actually it returns a function that takes dictionary of parameters as input (a requirement by TFA2 package).
    b_model     = model.get_binned_model(bin_scheme = bscheme, applyEff = effn, applyResponse = resn, eff_fname = effpath, res_fname = respath)
    #print(b_model())
    nll = model.binned_nll(b_data, b_model, gauss_constraint_ff = True)
    tot_params = model.tot_params #get both fixed and floated parameter dictionary
    print('NLL: ', nll(tot_params).numpy()) #print the nll value of the model given the binned data with the parameters with which the model was created

    #set the values of the parameters to the one passed in before fitting
    if fit_param_vals is not None:
        model.set_params_values(fit_param_vals, isfitresult = False)
        print('NLL after setting param vals: ', nll(tot_params).numpy()) 

    exit(1)
    #For a given data sample (b_data) one would conduct nfits with different starting values for the free parameters and use the results of the fit that gives the least nll value.
    #This will ensure that we are converging to a global minima. The default value of nfits is 1, so for now only one fit is conducted. 
    reslts = Minimize(nll, model, tot_params, nfits = nfits, use_hesse = usehesse, use_minos = useminos, use_grad = usegrad, randomiseFF = randomiseFF, get_covariance = get_covariance)
    #pprint.pprint(reslts)

    ##write the fit results to a text file
    resname = 'results_'+floatWC+'_'+str(seed)+'_'+bscheme+'_'+str(nevnts)+'_'+'_'.join(floated_FF)+'_'+suffix
    resfname = direc+resname+'.txt'
    print('resfname', resfname)
    model.write_fit_results(reslts, resfname, get_covariance = get_covariance)

    #plot the fit results
    plotfname = direc+resname+'.pdf'
    b_fit = b_model().numpy() #After the setting the parameter values to the estimated ones (as above), we get here the predicted shape
    if plotRes: model.plot_fitresult_binned_2D(b_data, b_fit, bin_scheme = bscheme, fname = plotfname)
    print('Final Nll:', nll(tot_params).numpy())
    
    end = time.time(); print('Time taken in min', (end - start)/60.)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for LbToLclnu_modeldependency.py')
    #required aguments
    parser.add_argument('-f', '--floatWC'   , dest='floatWC',type=str, required=True, help='(string) Name of the Wilson coefficient (WC) to be floated. Available options are [CVR,CSR,CSL,CT,None].')
    parser.add_argument('-s', '--seed'      , dest='seed'   ,type=int, required=True, help='(int) Seed for generation of fake/toy data. This should be different for each toy.')
    #optional arguments
    parser.add_argument('-b', '--bscheme'   , dest='bscheme',type=str, default='Scheme5',help='(string) Binning scheme to be used. Available options are [Scheme0,Scheme1,Scheme2,Scheme3,Scheme4,Scheme5,Scheme6] and default is Scheme5 (7 x 6). See BinningSchemes/Binning_Scheme.py.')
    parser.add_argument('-n', '--nevnts'    , dest='nevnts' ,type=int, default=int(7.5e6),help='(int) Size of the toy sample. Default is 7.5M events.')
    parser.add_argument('-nf','--nfits'     , dest='nfits'  ,type=int, default=1,help='(int) Number of fits to conduct to a given sample. Default in 1.')
    parser.add_argument('-sf','--suffix'    , dest='suffix' ,type=str, default='toy',help="(int) A unique suffix added to the name of the fit result file (*_suffix.txt) and plot file (*_suffix.pdf). Default is 'toy'.")
    parser.add_argument('-d', '--direc'     , dest='direc'  ,type=str, default='./plots/',help='(string) Directory in which the fit result (.txt) and plot is to be saved. Default in ./plots/.')
    parser.add_argument('-p', '--plotRes'   , dest='plotRes',type=str2bool,default='True',help='(bool) Set to False if you do not want to plot the result. Default is True.')
    parser.add_argument('-effn', '--effn'   , dest='effn'   ,type=str2bool,default='False',help='(bool) Set to True if you want efficiency included in model. Default is False.')
    parser.add_argument('-effp', '--effpath', dest='effpath',type=str, default=fitdir+'/Eff/Eff_Tot_SM_nominal.p',help='(string) Path to efficiency file. Default is: '+fitdir+'/Eff/Eff_Tot_SM_nominal.p')
    parser.add_argument('-resn', '--resn'   , dest='resn'   ,type=str2bool,default='False',help='(bool) Set to True if you want resolution information included in model. Default is False.')
    parser.add_argument('-resp', '--respath', dest='respath',type=str,default=fitdir+'/ResponseMatrix/responsematrix_nominal.p',help='(bool) Path to resoponse matrix file. Default is:'+fitdir+'/ResponseMatrix/responsematrix_nominal.p')
    parser.add_argument('-utgen', '--unbinned_toygen', dest='unbinned_toygen' ,type=str2bool,default='True',help='(bool) Set to True if you want toys generated in an unbinned manner and False if you want the bin content to be fluctuated with Poisson error. Default is True.')
    parser.add_argument('-e', '--floated_FF', dest='floated_FF',nargs='+', default = ['None'], 
    help="(list) List of form factor (FF) parameters that you want floated in the fit. \
          Default is 'None' that is all FF parameters are fixed. \
          When CVR or CSR or CSL is set as 'floatWC': 11 FF parameters can be floated, they are a0f0 a0fplus a0fperp a1f0 a1fplus a1fperp a0g0 a0gplus a1g0 a1gplus a1gperp \
          When CT is set as 'floatWC': In addition to the 11 FF, we can float 7 more which are a0hplus a0hperp a0htildeplus a1hplus a1hperp a1htildeplus a1htildeperp") 
    parser.add_argument('-inter', '--cpuinter'    , dest='cpuinter' ,type=int, default=int(1),help='(int) Number of cores to use for TensorFlow for INTER operations. Default is 1 core.')
    parser.add_argument('-intra', '--cpuintra'    , dest='cpuintra' ,type=int, default=int(1),help='(int) Number of cores to use for TensorFlow for INTRA operations (matrix multiplication, etc). Default is 1 core.')
    parser.add_argument('-g'    , '--usegrad'     , dest='usegrad'  ,type=str2bool,default='True',help='(bool) Set to True if you want TF gradients to be used instead of Minuit in minimisation. Default is True.')
    parser.add_argument('-uh'   , '--usehesse'   , dest='usehesse' ,type=str2bool,default='True',help='(bool) Set to False if you do want HESSE running after MIGRAD. Default is True.')
    parser.add_argument('-um'   , '--useminos'   , dest='useminos' ,type=str2bool,default='False',help='(bool) Set to True if you want MINOS running after HESSE. Default is False.')
    parser.add_argument('-rFF'  , '--randomiseFF' , dest='randomiseFF',type=str2bool,default='True',help='(bool) Set to True if you want FF randomised before fitting or else just set to the LQCD central values. Default is True.')
    parser.add_argument('-cov', '--get_covariance', dest='get_covariance',type=str2bool,default='False',help='Set to True, if you want to store the covariance matrix. Default is false.')
    parser.add_argument('-gpv', '--gen_param_vals', type=str,   help='(dict) Dictionary  of {param_name: param_value} to set before generating toy sample. Default is None i.e. SM values used.', default = 'None')
    parser.add_argument('-fpv', '--fit_param_vals', type=str,   help='(dict) Dictionary  of {param_name: param_value} to set before fitting sample (after generating). Default is None i.e. floated params are randomised and non-floated are fixed to SM values.' , default = 'None')
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
    cpuinter   = args.cpuinter
    cpuintra   = args.cpuintra
    unbinned_toygen = args.unbinned_toygen
    usegrad    = args.usegrad
    usehesse   = args.usehesse
    useminos   = args.useminos
    randomiseFF= args.randomiseFF
    get_covariance= args.get_covariance
    gen_param_vals = eval(args.gen_param_vals) #convert to dictionary
    fit_param_vals = eval(args.fit_param_vals) #convert to dictionary
    print('Arguments to the file are:')
    for arg in vars(args): print(arg,':',getattr(args, arg))
    #print(args)
    tf.config.threading.set_intra_op_parallelism_threads(cpuinter)
    tf.config.threading.set_inter_op_parallelism_threads(cpuintra)
    if not direc.endswith('/'): direc += '/'

    #make direc if it does not exist
    if not os.path.exists(direc):
        os.system('mkdir '+direc)

    #if unbinned_toygen is True then make directory where toy is stored
    if unbinned_toygen:
        if not os.path.exists(direc+'toyrootfiles'):
            print('Making directory since unbinned_toygen is set to True', direc+'toyrootfiles')
            os.system('mkdir '+direc+'toyrootfiles')
    
    main()
