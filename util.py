import tensorflow as tf
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU
import numpy as np
home = os.getenv('HOME')
import pickle
import tensorflow_probability as tfp
tfd = tfp.distributions
import functools
#amplitf
sys.path.append(home+"/Packages/AmpliTF/")
import amplitf.interface as atfi
import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace
#tfa2
sys.path.append(home+"/Packages/TFA2/")
import tfa.optimisation as tfo
import tfa.toymc as tft
#binning related
sys.path.append(os.path.abspath(os.getcwd()))
from BinningSchemes.Binning_Scheme import defing_binning_scheme
fitdir=os.path.dirname(os.path.abspath(__file__))
import pprint
from root_pandas import to_root
from root_pandas import read_root
import pandas as pd
import argparse

class LbToLclNu_Model:
    """Class that defines the Lb->Lclnu decay"""

    def __init__(self, MLb, MLc, Mlep, wc_floated_names = ['CVR'], ff_floated_names = ['None']):
        """initialise some variables"""
        #Masses of particles involved 
        self.MLb     = MLb
        self.MLc     = MLc
        self.Mlep    = Mlep
        #print(self.MLb, self.MLc, self.Mlep)
    
        #Constants expressed in GeV
        self.GF      = 1.166378e-5 #GeV^-2
        self.Vcb     = 4.22e-2     #avg of incl and excl
        #print(self.GF, self.Vcb)
        
        #Define function defined in lattice QCD here related to form factors
        self.Tf_plus = {} #mf_pole = M_BC + delta_f and Tf_plus = mf_pole**2
        self.Tf_plus['fplus']      = np.power(6.276 + 56e-3         , 2)#GeV
        self.Tf_plus['fperp']      = np.power(6.276 + 56e-3         , 2)#GeV
        self.Tf_plus['f0']         = np.power(6.276 + 449e-3        , 2)#GeV
        self.Tf_plus['gplus']      = np.power(6.276 + 492e-3        , 2)#GeV
        self.Tf_plus['gperp']      = np.power(6.276 + 492e-3        , 2)#GeV
        self.Tf_plus['g0']         = np.power(6.276 + 0.            , 2)#GeV
        self.Tf_plus['hplus']      = np.power(6.276 + 6.332 - 6.276 , 2)#GeV
        self.Tf_plus['hperp']      = np.power(6.276 + 6.332 - 6.276 , 2)#GeV
        self.Tf_plus['htildeplus'] = np.power(6.276 + 6.768 - 6.276 , 2)#GeV
        self.Tf_plus['htildeperp'] = np.power(6.276 + 6.768 - 6.276 , 2)#GeV
        #print(self.Tf_plus)
        
        #Map to relate W* resonance helicity to the sign in the amplitude (used in building the amplitde)
        self.eta    = {'t': 1.,  0 : -1.,  -1: -1., 1 : -1.}
        #print(self.eta)
        
        #Map to relate WC to the sign in the amplitude (used in building the amplitde)
        self.signWC = {'V': 1., 'A': -1., 'S': 1., 'PS': -1., 'T': 1., 'PT': -1.}
        #print(self.signWC)

        #make SM wc values
        self.wc_sm = {}
        self.wc_sm['CVL'] = 0.
        self.wc_sm['CVL'] = 0.
        self.wc_sm['CVR'] = 0.
        self.wc_sm['CSR'] = 0.
        self.wc_sm['CSL'] = 0.
        self.wc_sm['CT']  = 0.
        #print(self.wc_sm)
        
        #make SM FF mean and covariance values 
        #ff mean
        self.ff_mean  = {}
        f     = open(fitdir+"/FF_cov/LambdabLambdac_results.dat", "r")
        for l in f.readlines(): self.ff_mean[l.split()[0]] = float(l.split()[1])
        f.close()
        #print(self.ff_mean)

        #ff covariant
        self.ff_cov = {}
        for l in list(self.ff_mean.keys()): self.ff_cov[l] = {}
        f  = open(fitdir+"/FF_cov/LambdabLambdac_covariance.dat", "r")
        for l in f.readlines(): self.ff_cov[l.split()[0]][l.split()[1]] = float(l.split()[2])
        f.close()
        #print(self.ff_cov)
        
        #Lists that will serve as index for incoherent sum of amplitudes (defined in units of 1/2 except for lw used in leptonic case that does not need this)
        self.lLbs = [1, -1]
        self.lLcs = [1, -1]
        self.lls  = [1, -1]
        #print(self.lLbs, self.lLcs, self.lls)
        
        #Lists that will serve as index for coherent sum of amplitudes (defined in units of 1/2 except for lw used in leptonic case that does not need this)
        self.lws  = ['t', 0, 1, -1]
        self.WCs  = ['V', 'A', 'S', 'PS', 'T', 'PT']
        self.FFs  = ['a0f0', 'a0fplus', 'a0fperp', 'a0g0', 'a0gplus', 'a0hplus', 'a0hperp', 'a0htildeplus']
        self.FFs += ['a1f0', 'a1fplus', 'a1fperp', 'a1g0', 'a1gplus', 'a1gperp', 'a1hplus', 'a1hperp', 'a1htildeplus', 'a1htildeperp']
        self.FFs += ['a0gperp', 'a0htildeperp']
        #print(self.lws, self.WCs, self.FFs)
        
        #Map defining WC -> FF (Used to turn off FF floating if WC is not floating)
        self.map_wc_ff = {}
        self.map_wc_ff['SM']   = ['a0f0', 'a0fplus', 'a0fperp', 'a1f0', 'a1fplus', 'a1fperp', 'a0g0', 'a0gplus', 'a0gperp', 'a1g0', 'a1gplus', 'a1gperp']
        self.map_wc_ff['CVL']  = ['a0f0', 'a0fplus', 'a0fperp', 'a1f0', 'a1fplus', 'a1fperp', 'a0g0', 'a0gplus', 'a0gperp', 'a1g0', 'a1gplus', 'a1gperp']
        self.map_wc_ff['CVR']  = ['a0f0', 'a0fplus', 'a0fperp', 'a1f0', 'a1fplus', 'a1fperp', 'a0g0', 'a0gplus', 'a0gperp', 'a1g0', 'a1gplus', 'a1gperp']
        self.map_wc_ff['CSL']  = ['a0f0', 'a1f0', 'a0g0', 'a1g0']
        self.map_wc_ff['CSR']  = ['a0f0', 'a1f0', 'a0g0', 'a1g0']
        self.map_wc_ff['CT']   = ['a0hplus', 'a0hperp', 'a0htildeperp', 'a0htildeplus', 'a1hplus', 'a1hperp', 'a1htildeplus', 'a1htildeperp']
        #print(self.map_wc_ff)
    
        #get wilson coefficients (wc)
        self.wc_params     = self.get_wc_params()
        #get form factor (ff) parameters
        self.ff_params     = self.get_ff_params()
        #define dictionary with all parms (fixed and floated)
        self.tot_params    = {**self.wc_params, **self.ff_params}
        #define limits and rectangular phase space i.e. (qsq, costhmu) space
        phsp_limits  = [(self.Mlep**2, (self.MLb - self.MLc)**2)] 
        phsp_limits += [(-1., 1.)]
        self.phase_space = RectangularPhaseSpace(ranges=phsp_limits)

        #float only the necesarry wilson coefficient
        self.fix_or_float_params(wc_floated_names)
        self.wc_floated = [p for p in list(self.wc_params.values()) if p.floating()]
        self.wc_floated_names = [p.name for p in self.wc_floated]
        if len(self.wc_floated) == 0:
            self.wc_floated = None
            self.wc_floated_names = None

        #float only the necesarry ff params
        self.ff_floated = None
        self.ff_floated_names = None
        self.ff_floated_mean  = None
        self.ff_floated_cov   = None
        if ff_floated_names[0] == 'None':
            print('All form factors parameters are fixed!')
        elif ff_floated_names[0] == 'All':
            print('All form factors parameters related to the WC are floated!')
            self.fix_or_float_params(self.map_wc_ff['SM'])
            if self.wc_floated_names is not None:
                if 'CT' in self.wc_floated_names:
                    self.fix_or_float_params(self.map_wc_ff['CT'])

            self.ff_floated = [p for p in list(self.ff_params.values()) if p.floating()]
            self.ff_floated_names = [p.name for p in self.ff_floated]
            self.ff_floated_mean, self.ff_floated_cov = self.get_FF_mean_cov()
        else:
            self.fix_or_float_params(ff_floated_names)
            self.ff_floated = [p for p in list(self.ff_params.values()) if p.floating()]
            self.ff_floated_names = [p.name for p in self.ff_floated]
            self.ff_floated_mean, self.ff_floated_cov = self.get_FF_mean_cov()

    def get_wc_params(self):
        """Make WC parameters"""
        Wlcoef = {}
        Wlcoef['CVL'] = tfo.FitParameter("CVL" , 0.0, -2., 2., 0.08)
        Wlcoef['CVR'] = tfo.FitParameter("CVR" , 0.0, -2., 2., 0.08)
        Wlcoef['CSR'] = tfo.FitParameter("CSR" , 0.0, -2., 2., 0.08)
        Wlcoef['CSL'] = tfo.FitParameter("CSL" , 0.0, -2., 2., 0.08)
        Wlcoef['CT']  = tfo.FitParameter("CT"  , 0.0, -2., 2., 0.08)
        for k in list(Wlcoef.keys()): Wlcoef[k].fix() #Fix all the wcs
        print('Setting WCs to SM value of zero and allowed to vary in the fit b/w [-2., 2.]')
        return Wlcoef

    def get_ff_params(self):
        """Make FF parameters"""
        #make dictionary of ff params
        ffact = {}
        print('Setting FF to LQCD central value and allowed to vary b/w [lqcd_val - 20 * lqcd_sigma, lqcd_val + 20 * lqcd_sigma]. So this corresponds to...')
        for FF in self.FFs[:-2]: 
            ff_mn  = self.ff_mean[FF]
            ff_sg  = np.sqrt(self.ff_cov[FF][FF])
            ff_l   = ff_mn - 20.*ff_sg
            ff_h   = ff_mn + 20.*ff_sg
            print('Setting', FF, 'to SM value:', ff_mn, 'with sigma', ff_sg, ', allowed to vary in fit b/w [', ff_l, ff_h, ']')
            ffact[FF] = tfo.FitParameter(FF, ff_mn, ff_l, ff_h, 0.08)
            ffact[FF].fix() #fix all the ff params
            
        return ffact

    def fix_or_float_params(self, fp_list, fix = False, verbose = True):
        """Given a list of parmeter names float of fix the paramters"""
        for p in list(self.tot_params.values()):
            if p.name in fp_list:
                if fix:
                    if verbose: print('Fixing', p.name)
                    p.fix()
                else:
                    if verbose: print('Floating', p.name)
                    p.float()

    def get_FF_mean_cov(self):
        """get the FF mean and covariant for FF parameters""" 
    
        #define a check for pos definite covariant matrix
        def _is_symm_pos_def(A):

            def _check_symmetric(a, rtol=1e-05, atol=1e-08): 
                return np.allclose(a, a.T, rtol=rtol, atol=atol)
        
            if _check_symmetric(A):
                try:
                    np.linalg.cholesky(A)
                    return True
                except np.linalg.LinAlgError:
                    return False
            else:
                return False
    
        if self.ff_floated_names is None:
            raise Exception("Nothing is floated to cannot get mean and covariance for floated FF params.")

        #get FF parameter  that are floated
        #get mean
        mean_list = [self.ff_mean[n] for n in self.ff_floated_names]
        mean_list = np.array(mean_list)
        #get cov
        cov_list  = [[self.ff_cov[l1][l2] for l2 in self.ff_floated_names] for l1 in self.ff_floated_names]
        cov_list  = np.array(cov_list)
        #check if symmetric and positive definite
        if not _is_symm_pos_def(cov_list): 
            print('Not symmetric positive definite cov matrix, exiting')
            exit(0)
        else:
            print('Cov is symmetric and positive definite')
    
        return mean_list, cov_list

    def get_freeparams(self):
        """Make dictionary of free parameters i.e. WC * FF"""

        #define function for change of basis
        def _wc_basis(basis_name):
            """Change the WC basis e.g. 1+CVL+CVR -> V"""
            wcp = None
            if basis_name == 'V':
                wcp =  atfi.const(1.) + self.wc_params['CVL']() + self.wc_params['CVR']()
            elif basis_name == 'A':
                wcp =  atfi.const(1.) + self.wc_params['CVL']() - self.wc_params['CVR']()
            elif basis_name == 'S':
                wcp =  self.wc_params['CSL']() + self.wc_params['CSR']()
            elif basis_name == 'PS':
                wcp =  self.wc_params['CSL']() - self.wc_params['CSR']()
            elif basis_name == 'T' or basis_name == 'PT':
                wcp =  self.wc_params['CT']()
            else:
                raise Exception('The passed basis_name', basis_name, 'not recognised')
    
            return wcp

        def _ff_common(ff_name):
            """Set a0gplus = a0gperp and a0htildeplus = a0htildeperp"""
            ffp = None
            if ff_name == 'a0gperp':
                ffp = self.ff_params['a0gplus']()
            elif ff_name == 'a0htildeperp':
                ffp = self.ff_params['a0htildeplus']()
            else:
                ffp = self.ff_params[ff_name]()

            return ffp

        #Define free parameters dictionary
        free_params = {}
        for WC in self.WCs: 
            for FF in self.FFs:
                free_params[( WC, FF)] = _wc_basis(WC) * _ff_common(FF)
    
        #print(len(free_params.keys()))                        #6*20 = 120
        return free_params

    def prepare_data(self,x):
        """Function that calculates most of the variables for phase space variables"""
        Vars = {}
        #Store Lc phase_space Varsiables and make sure volume element stays as dq2*dcthlc*dcthl*dphl*dm2pk
        q2            = x[:,0]
        w_ctheta_l    = x[:,1] #This because in MC it is like this
        lb_ctheta_lc  = atfi.ones(x[:,0])  #Lb polarisation is zero (shape same as integrating or fixing)
        lb_phi_lc     = atfi.zeros(x[:,0]) #always zeros
        w_phi_l       = atfi.zeros(x[:,0]) #Lb polarisation is zero (shape same as integrating or fixing)
    
        #Store info of other particles
        lb_ctheta_w   = -lb_ctheta_lc
        lb_phi_w      =  lb_phi_lc+atfi.pi()
        w_ctheta_nu   = -w_ctheta_l
        w_phi_nu      =  w_phi_l+atfi.pi()
    
        #Lc and W 4mom in Lb rest frame     (zLb = pbeam_lab x pLb_lab, yLb = zLb x lb_p3vec_lc, x = y x z => r, theta, phi = lb_p3mag_lc,lb_ctheta_lc, 0)
        lb_p3mag_lc  = pvecMag(atfi.const(self.MLb)**2, atfi.const(self.MLc)**2, q2)
        lb_p4mom_lc  = MakeFourVector(lb_p3mag_lc,     lb_ctheta_lc, lb_phi_lc, atfi.const(self.MLc)**2)
        lb_p4mom_w   = MakeFourVector(lb_p3mag_lc,     lb_ctheta_w,  lb_phi_w , q2)
        #l and nu 4mom in W helicity frame (zw = lb_p3vec_w, yw = yLb = zLb x lb_p3vec_lc , xw = yw x zw => r, theta, phi => w_p3mag_l, w_ctheta_l, w_phl_l)
        w_p3mag_l    = pvecMag(q2, atfi.const(self.Mlep)**2, 0.)
        w_p4mom_l    = MakeFourVector(w_p3mag_l,    w_ctheta_l , w_phi_l , atfi.const(self.Mlep)**2)
        w_p4mom_nu   = MakeFourVector(w_p3mag_l,    w_ctheta_nu, w_phi_nu, 0.)
    
        #Get everything is Lb rest frame
        lb_p4mom_l , lb_p4mom_nu = InvRotateAndBoostFromRest(w_p4mom_l,   w_p4mom_nu, lb_phi_w,  lb_ctheta_w,  lb_p4mom_w)
    
        #Store Varss
        Vars['q2']            = q2
        Vars['El']            = atfk.time_component(lb_p4mom_l)
        Vars['m_Lbmu']        = atfk.mass(lb_p4mom_lc+lb_p4mom_l)
        #lnu angles
        Vars['lb_ctheta_w']   = lb_ctheta_w
        Vars['lb_phi_w']      = lb_phi_w
        Vars['w_ctheta_l']    = w_ctheta_l
        Vars['w_phi_l']       = w_phi_l
        #3mom mag 
        Vars['lb_p3mag_lc']   = lb_p3mag_lc
        Vars['w_p3mag_l']     = w_p3mag_l
    
        return Vars

    def d_phi_twobody(self,mijsq, misq, mjsq):
        """Two body phase space element"""
        return 1./(2.**4 * atfi.pi()**2) * pvecMag(mijsq, misq, mjsq)/atfi.sqrt(mijsq)

    def get_phasespace_term(self, Obs):
        """Two body phase space factors along with the element"""
        #d(lb_ctheta_lc) * d(q2) * d(w_ctheta_l) * d(w_phi_l)
        phsspace  = 1./(2. * atfi.const(self.MLb) * 2. * atfi.pi()) * self.d_phi_twobody(atfi.const(self.MLb)**2, atfi.const(self.MLc)**2, Obs['q2'])   #Lb -> Lc W
        phsspace *= self.d_phi_twobody(Obs['q2'], atfi.const(self.Mlep)**2, 0.)   #W  ->  l nu
        return phsspace

    def get_lb_ampls(self, Obs):
        """
        Some important comments about the hadronic currents:
        V currents:
            - aV, bV and cV are form factor parameter independent (unlike in our paper)
            - When summing over lwd only one term is needed. This term is taken as 't' (b'cos eta[t] = 1). Make sure Leptonic V and A current lwd='t'.
            - The ffterms func return corrent q2 dependent terms pertaining to FF parameter a0 and a1.
            - Hadr(lb, lc, lw, 't', 'V', 'a0') have 16 comb out of which 12 survive and 4 are zero
        S currents:
            - Fake lw and lwd index (see above for reason). Fixed to zero, make sure Leptonic S, PS currents also have lw = lwd = 't'.
            - Hadr(lb, lc, 't', 't', 'S', 'a0') have 4 terms all which are nonzero.
        A currents: 
            - Like in Leptonic current these are NOT equal to V currents.
            - Same comments as V currents.
        PS currents: 
            - Like in Leptonic current these are NOT equal to S currents.
            - Same comments as S currents.
        T currents:
            - Hadr(lb, lc, lw, lwd, 'T', 'a0') have 64 terms out of which 32 are nonzero.
        PT currents:
            - Like in Leptonic current these are NOT equal to T currents.
            - same comments as T currents.
        Func returns:
            Dict with index as tuples Hadr[(lb, lc, lw, lwd, wc, ff)]
        """
        #q2 and angular terms: Used sin(x/2) = sqrt((1-cthlc)/2) and cos(x/2)=sqrt((1+cthlc)/2).Since x/2 ranges from 0 to pi/2 both sine and cosine are pos.
        q2          =  Obs['q2']
        Mpos        = atfi.const(self.MLb) + atfi.const(self.MLc)
        Mneg        = atfi.const(self.MLb) - atfi.const(self.MLc)
        sqrtQp      = atfi.sqrt(atfi.pow(Mpos,2) - q2)
        sqrtQn      = atfi.sqrt(atfi.pow(Mneg,2) - q2)
        sqrtQpMn    = sqrtQp * Mneg
        sqrtQnMp    = sqrtQn * Mpos
        aV          = sqrtQpMn/atfi.sqrt(q2)
        bV          = sqrtQnMp/atfi.sqrt(q2)
        cV          = atfi.sqrt(np.array(2.)) * sqrtQn
        aA          = bV
        bA          = aV
        cA          = atfi.sqrt(np.array(2.)) * sqrtQp
        mb          = atfi.const(4.18       ) #GeV
        mc          = atfi.const(1.275      ) #GeV
        aS          = sqrtQpMn/(mb - mc)
        aP          = sqrtQnMp/(mb + mc)
        aT          = sqrtQn
        bT          = atfi.sqrt(np.array(2.)) * bV
        cT          = atfi.sqrt(np.array(2.)) * aV
        dT          = sqrtQp
        cthlc       = -Obs['lb_ctheta_w'] #ONE Lb polarisation ignored
        costhlchalf = atfi.sqrt((1+cthlc)/2.)  #One Lb polarisation ignored
        sinthlchalf = atfi.sqrt((1-cthlc)/2.)  #ZERO Lb polarisation ignored, so commented out the terms
        t0          = atfi.pow(Mneg,2)
    
        #get only q2 dependent terms in FF expansion i.e. ones pertaining to a0 and a1.
        ffterms = {}
        for ff in ['fplus', 'fperp', 'f0', 'gplus', 'gperp', 'g0', 'hplus', 'hperp', 'htildeperp', 'htildeplus']:
            cf = 1./(1. - q2/atfi.const(self.Tf_plus[ff]))
            zf = (atfi.sqrt(atfi.const(self.Tf_plus[ff]) - q2) - atfi.sqrt(atfi.const(self.Tf_plus[ff]) - t0))/(atfi.sqrt(atfi.const(self.Tf_plus[ff]) - q2) + atfi.sqrt(atfi.const(self.Tf_plus[ff]) - t0))
            ffterms['a0'+ff] =  cf
            ffterms['a1'+ff] =  cf * zf
    
        #define hadronic amplitudes
        Hadr = {}
        #f0 terms: contains V and S currents
        for a in ['a0f0', 'a1f0']:
            Hadr[( 1, 1,'t', 't', 'V', a)] =  costhlchalf * aV * ffterms[a]
            Hadr[(-1,-1,'t', 't', 'V', a)] =  costhlchalf * aV * ffterms[a]
            Hadr[( 1, 1,'t', 't', 'S', a)] =  costhlchalf * aS * ffterms[a]
            Hadr[(-1,-1,'t', 't', 'S', a)] =  costhlchalf * aS * ffterms[a]
            #Hadr[( 1,-1,'t', 't', 'V', a)] = -sinthlchalf * aV * ffterms[a]
            #Hadr[(-1, 1,'t', 't', 'V', a)] =  sinthlchalf * aV * ffterms[a]
            #Hadr[( 1,-1,'t', 't', 'S', a)] = -sinthlchalf * aS * ffterms[a]
            #Hadr[(-1, 1,'t', 't', 'S', a)] =  sinthlchalf * aS * ffterms[a]
        
        #fplus terms
        for a in ['a0fplus', 'a1fplus']:
            Hadr[( 1, 1,  0, 't', 'V', a)] =  costhlchalf  * bV * ffterms[a]
            Hadr[(-1,-1,  0, 't', 'V', a)] =  costhlchalf  * bV * ffterms[a]
            #Hadr[( 1,-1,  0, 't', 'V', a)] = -sinthlchalf  * bV * ffterms[a]
            #Hadr[(-1, 1,  0, 't', 'V', a)] =  sinthlchalf  * bV * ffterms[a]
        
        #fperp terms
        for a in ['a0fperp', 'a1fperp']:
            Hadr[( 1,-1, -1, 't', 'V', a)] = -costhlchalf  * cV * ffterms[a]
            Hadr[(-1, 1,  1, 't', 'V', a)] = -costhlchalf  * cV * ffterms[a]
            #Hadr[(-1,-1, -1, 't', 'V', a)] = -sinthlchalf  * cV * ffterms[a]
            #Hadr[( 1, 1,  1, 't', 'V', a)] =  sinthlchalf  * cV * ffterms[a]
        
        #g0 terms: contains A and PS currents
        for a in ['a0g0','a1g0']:
            Hadr[( 1, 1,'t', 't', 'A', a)] =  costhlchalf * aA * ffterms[a]
            Hadr[(-1,-1,'t', 't', 'A', a)] = -costhlchalf * aA * ffterms[a]
            Hadr[( 1, 1,'t', 't','PS', a)] = -costhlchalf * aP * ffterms[a]
            Hadr[(-1,-1,'t', 't','PS', a)] =  costhlchalf * aP * ffterms[a]
            #Hadr[( 1,-1,'t', 't', 'A', a)] =  sinthlchalf * aA * ffterms[a]
            #Hadr[(-1, 1,'t', 't', 'A', a)] =  sinthlchalf * aA * ffterms[a]
            #Hadr[( 1,-1,'t', 't','PS', a)] = -sinthlchalf * aP * ffterms[a]
            #Hadr[(-1, 1,'t', 't','PS', a)] = -sinthlchalf * aP * ffterms[a]
        
        #gplus terms
        for a in ['a0gplus','a1gplus']:
            Hadr[( 1, 1,  0, 't', 'A', a)] =  costhlchalf * bA * ffterms[a]
            Hadr[(-1,-1,  0, 't', 'A', a)] = -costhlchalf * bA * ffterms[a]
            #Hadr[( 1,-1,  0, 't', 'A', a)] =  sinthlchalf * bA * ffterms[a]
            #Hadr[(-1, 1,  0, 't', 'A', a)] =  sinthlchalf * bA * ffterms[a]
        
        #gperp terms
        for a in ['a0gperp','a1gperp']:
            Hadr[( 1,-1, -1, 't', 'A', a)] =  costhlchalf * cA * ffterms[a]
            Hadr[(-1, 1,  1, 't', 'A', a)] = -costhlchalf * cA * ffterms[a]
            #Hadr[(-1,-1, -1, 't', 'A', a)] =  sinthlchalf * cA * ffterms[a]
            #Hadr[( 1, 1,  1, 't', 'A', a)] =  sinthlchalf * cA * ffterms[a]
        
        #hplus terms: T and PT
        for a in ['a0hplus','a1hplus']:
            Hadr[( 1, 1,'t',  0, 'T', a)] =  costhlchalf * aT * ffterms[a]
            Hadr[(-1,-1,'t',  0, 'T', a)] =  costhlchalf * aT * ffterms[a]
            Hadr[( 1, 1,  0,'t', 'T', a)] = -costhlchalf * aT * ffterms[a]
            Hadr[(-1,-1,  0,'t', 'T', a)] = -costhlchalf * aT * ffterms[a]
            Hadr[( 1, 1,  1, -1, 'PT',a)] =  costhlchalf * aT * ffterms[a]
            Hadr[(-1,-1,  1, -1, 'PT',a)] =  costhlchalf * aT * ffterms[a]
            Hadr[( 1, 1, -1,  1, 'PT',a)] = -costhlchalf * aT * ffterms[a]
            Hadr[(-1,-1, -1,  1, 'PT',a)] = -costhlchalf * aT * ffterms[a]
            #Hadr[( 1,-1,'t',  0, 'T', a)] = -sinthlchalf * aT * ffterms[a]
            #Hadr[(-1, 1,'t',  0, 'T', a)] =  sinthlchalf * aT * ffterms[a]
            #Hadr[( 1,-1,  0,'t', 'T', a)] =  sinthlchalf * aT * ffterms[a]
            #Hadr[(-1, 1,  0,'t', 'T', a)] = -sinthlchalf * aT * ffterms[a]
            #Hadr[( 1,-1,  1, -1, 'PT',a)] = -sinthlchalf * aT * ffterms[a]
            #Hadr[(-1, 1,  1, -1, 'PT',a)] =  sinthlchalf * aT * ffterms[a]
            #Hadr[( 1,-1, -1,  1, 'PT',a)] =  sinthlchalf * aT * ffterms[a]
            #Hadr[(-1, 1, -1,  1, 'PT',a)] = -sinthlchalf * aT * ffterms[a]
        
        #hperp terms: T and PT
        for a in ['a0hperp','a1hperp']:
            Hadr[(-1, 1,'t',  1, 'T', a)] = -costhlchalf * bT *  ffterms[a]
            Hadr[( 1,-1,'t', -1, 'T', a)] = -costhlchalf * bT *  ffterms[a]
            Hadr[(-1, 1,  1,'t', 'T', a)] =  costhlchalf * bT *  ffterms[a]
            Hadr[( 1,-1, -1,'t', 'T', a)] =  costhlchalf * bT *  ffterms[a]
            Hadr[( 1,-1,  0, -1, 'PT',a)] = -costhlchalf * bT *  ffterms[a]
            Hadr[( 1,-1, -1,  0, 'PT',a)] =  costhlchalf * bT *  ffterms[a]
            Hadr[(-1, 1,  0,  1, 'PT',a)] =  costhlchalf * bT *  ffterms[a]
            Hadr[(-1, 1,  1,  0, 'PT',a)] = -costhlchalf * bT *  ffterms[a]
            #Hadr[(-1,-1,'t', -1, 'T', a)] = -sinthlchalf * bT *  ffterms[a]
            #Hadr[( 1, 1,'t',  1, 'T', a)] =  sinthlchalf * bT *  ffterms[a]
            #Hadr[(-1,-1, -1,'t', 'T', a)] =  sinthlchalf * bT *  ffterms[a]
            #Hadr[( 1, 1,  1,'t', 'T', a)] = -sinthlchalf * bT *  ffterms[a]
            #Hadr[(-1,-1,  0, -1, 'PT',a)] = -sinthlchalf * bT *  ffterms[a]
            #Hadr[( 1, 1,  1,  0, 'PT',a)] =  sinthlchalf * bT *  ffterms[a]
            #Hadr[(-1,-1, -1,  0, 'PT',a)] =  sinthlchalf * bT *  ffterms[a]
            #Hadr[( 1, 1,  0,  1, 'PT',a)] = -sinthlchalf * bT *  ffterms[a]
        
        #htildeperp terms: T and PT
        for a in ['a0htildeperp','a1htildeperp']:
            Hadr[(-1, 1,  0,  1, 'T', a)] = -costhlchalf * cT * ffterms[a]
            Hadr[( 1,-1,  0, -1, 'T', a)] = -costhlchalf * cT * ffterms[a]
            Hadr[(-1, 1,  1,  0, 'T', a)] =  costhlchalf * cT * ffterms[a]
            Hadr[( 1,-1, -1,  0, 'T', a)] =  costhlchalf * cT * ffterms[a]
            Hadr[( 1,-1,'t', -1, 'PT',a)] = -costhlchalf * cT * ffterms[a]
            Hadr[(-1, 1,'t',  1, 'PT',a)] =  costhlchalf * cT * ffterms[a]
            Hadr[( 1,-1, -1,'t', 'PT',a)] =  costhlchalf * cT * ffterms[a]
            Hadr[(-1, 1,  1,'t', 'PT',a)] = -costhlchalf * cT * ffterms[a]
            #Hadr[(-1,-1,  0, -1, 'T', a)] = -sinthlchalf * cT * ffterms[a]
            #Hadr[( 1, 1,  0,  1, 'T', a)] =  sinthlchalf * cT * ffterms[a]
            #Hadr[(-1,-1, -1,  0, 'T', a)] =  sinthlchalf * cT * ffterms[a]
            #Hadr[( 1, 1,  1,  0, 'T', a)] = -sinthlchalf * cT * ffterms[a]
            #Hadr[( 1, 1,'t',  1, 'PT',a)] = -sinthlchalf * cT * ffterms[a]
            #Hadr[(-1,-1,'t', -1, 'PT',a)] = -sinthlchalf * cT * ffterms[a]
            #Hadr[( 1, 1,  1,'t', 'PT',a)] =  sinthlchalf * cT * ffterms[a]
            #Hadr[(-1,-1, -1,'t', 'PT',a)] =  sinthlchalf * cT * ffterms[a]
        
        #htildeplus terms: T and PT
        for a in ['a0htildeplus','a1htildeplus']:
            Hadr[( 1, 1,  1, -1, 'T', a)] = -costhlchalf * dT * ffterms[a]
            Hadr[(-1,-1,  1, -1, 'T', a)] =  costhlchalf * dT * ffterms[a]
            Hadr[( 1, 1, -1,  1, 'T', a)] =  costhlchalf * dT * ffterms[a]
            Hadr[(-1,-1, -1,  1, 'T', a)] = -costhlchalf * dT * ffterms[a]
            Hadr[( 1, 1,'t',  0, 'PT',a)] = -costhlchalf * dT * ffterms[a]
            Hadr[(-1,-1,'t',  0, 'PT',a)] =  costhlchalf * dT * ffterms[a]
            Hadr[( 1, 1,  0,'t', 'PT',a)] =  costhlchalf * dT * ffterms[a]
            Hadr[(-1,-1,  0,'t', 'PT',a)] = -costhlchalf * dT * ffterms[a]
            #Hadr[( 1,-1,  1, -1, 'T', a)] = -sinthlchalf * dT * ffterms[a]
            #Hadr[(-1, 1,  1, -1, 'T', a)] = -sinthlchalf * dT * ffterms[a]
            #Hadr[( 1,-1, -1,  1, 'T', a)] =  sinthlchalf * dT * ffterms[a]
            #Hadr[(-1, 1, -1,  1, 'T', a)] =  sinthlchalf * dT * ffterms[a]
            #Hadr[( 1,-1,'t',  0, 'PT',a)] = -sinthlchalf * dT * ffterms[a]
            #Hadr[(-1, 1,'t',  0, 'PT',a)] = -sinthlchalf * dT * ffterms[a]
            #Hadr[( 1,-1,  0,'t', 'PT',a)] =  sinthlchalf * dT * ffterms[a]
            #Hadr[(-1, 1,  0,'t', 'PT',a)] =  sinthlchalf * dT * ffterms[a]
    
        return Hadr

    def get_w_ampls(self, Obs):
        """
        Some important comments on all the leptonic currents
        V or A current:
            - Terms with WC index V and A are equal.
            - Since only one term appears in the ampl for lwd. We fix lwd to 't' as done in hadronic V, A currents. Index 't' choosen because eta['t'] = 1.
            - Therefore Lept(lw, 't', 'V', ll) has total 8 components out of which 1 is zero.
        S or PS current:
            - Terms with WC index S and PS are equal.
            - In this case both lw and lwd are fake indeces (see above for reason), choosen to be 't' (see above) as done in hadronic S, PS currents.
            - Lept('t', 't', 'S', ll) has total 2 components out of which 1 is zero.
        T or PT current:
            - Terms with WC index T and PT are equal.
            - Lept(lw,lwd,'T',l) has total 32 components out of which 8 are zero.
        Func returns:
            Dict with index as tuples Lept[(lw, lwd, wc, ll)]
        """
        
        #q2 and angular terms
        q2              = Obs['q2'] #real
        cthl            =-Obs['w_ctheta_l'] #real (our pdf is expressed a function of w_ctheta_nu not w_ctheta_l
        phl             = Obs['w_phi_l'] #ZERO Lb polarisation ignored as a result amplitude real
        v               = atfi.sqrt(1. - atfi.pow(atfi.const(self.Mlep),2)/q2) #real
        sinthl          = atfi.sqrt(1. - atfi.pow(cthl,2.)) #real
        expPlusOneIphl  = atfi.cos(phl)    #real and One since Lb polarisation ignored
        expMinusOneIphl = atfi.cos(-phl)   #real and One since Lb polarisation ignored
        expMinusTwoIphl = atfi.cos(-2.*phl)#real and One since Lb polarisation ignored
        OneMinuscthl    = (1. - cthl)
        OnePluscthl     = (1. + cthl)
        al              = 2. * atfi.const(self.Mlep) * v
        bl              = 2. * atfi.sqrt(q2)* v
        complexsqrttwo  = atfi.sqrt(np.array(2.))
        
        Lept = {}
        
        Lept[('t', 't', 'V',  1)] =  expMinusOneIphl * al
        Lept[(  0, 't', 'V',  1)] = -expMinusOneIphl * cthl * al
        Lept[(  1, 't', 'V',  1)] =  expMinusTwoIphl * sinthl * al/complexsqrttwo
        Lept[( -1, 't', 'V',  1)] = -sinthl * al/complexsqrttwo
        Lept[(  0, 't', 'V', -1)] =  sinthl * bl
        Lept[(  1, 't', 'V', -1)] =  expMinusOneIphl * OnePluscthl  * bl/complexsqrttwo
        Lept[( -1, 't', 'V', -1)] =  expPlusOneIphl  * OneMinuscthl * bl/complexsqrttwo
        
        Lept[('t', 't', 'S', 1)]  =  expMinusOneIphl * bl
        
        Lept[('t',  0, 'T', -1)]  =  sinthl * al
        Lept[(  1, -1, 'T', -1)]  =  sinthl * al
        Lept[(  0, -1, 'T', -1)]  =  expPlusOneIphl  * OneMinuscthl * al/complexsqrttwo
        Lept[('t', -1, 'T', -1)]  =  expPlusOneIphl  * OneMinuscthl * al/complexsqrttwo
        Lept[('t',  1, 'T', -1)]  =  expMinusOneIphl * OnePluscthl  * al/complexsqrttwo
        Lept[(  0,  1, 'T', -1)]  = -expMinusOneIphl * OnePluscthl  * al/complexsqrttwo
        Lept[('t',  0, 'T',  1)]  = -expMinusOneIphl * cthl * bl
        Lept[(  1, -1, 'T',  1)]  = -expMinusOneIphl * cthl * bl
        Lept[('t',  1, 'T',  1)]  =  expMinusTwoIphl * sinthl * bl/complexsqrttwo
        Lept[(  0,  1, 'T',  1)]  = -expMinusTwoIphl * sinthl * bl/complexsqrttwo
        Lept[('t', -1, 'T',  1)]  = -sinthl * bl/complexsqrttwo
        Lept[(  0, -1, 'T',  1)]  = -sinthl * bl/complexsqrttwo
        Lept[(  0,'t', 'T', -1)]  = -sinthl * al
        Lept[( -1,  1, 'T', -1)]  = -sinthl * al
        Lept[( -1,  0, 'T', -1)]  = -expPlusOneIphl  * OneMinuscthl * al/complexsqrttwo
        Lept[( -1,'t', 'T', -1)]  = -expPlusOneIphl  * OneMinuscthl * al/complexsqrttwo
        Lept[(  1,'t', 'T', -1)]  = -expMinusOneIphl * OnePluscthl  * al/complexsqrttwo
        Lept[(  1,  0, 'T', -1)]  =  expMinusOneIphl * OnePluscthl  * al/complexsqrttwo
        Lept[(  0,'t', 'T',  1)]  =  expMinusOneIphl * cthl * bl
        Lept[( -1,  1, 'T',  1)]  =  expMinusOneIphl * cthl * bl
        Lept[(  1,'t', 'T',  1)]  = -expMinusTwoIphl * sinthl * bl/complexsqrttwo
        Lept[(  1,  0, 'T',  1)]  =  expMinusTwoIphl * sinthl * bl/complexsqrttwo
        Lept[( -1,'t', 'T',  1)]  =  sinthl * bl/complexsqrttwo
        Lept[( -1,  0, 'T',  1)]  =  sinthl * bl/complexsqrttwo
    
        return Lept

    def get_dynamic_term(self, obsv):
        """Get the dynamic density term in Lb->Lclnu decay"""
    
        #define a helpful function
        def _replace_wc(strg):
            """Helpful function when for leptonic amplitudes in definition of density function"""
            strg_wc = strg.replace('A' , 'V')
            strg_wc = strg_wc.replace('PS', 'S')
            strg_wc = strg_wc.replace('PT', 'T')
            return strg_wc
    
        #get Lb and w amplitudes
        free_params   = self.get_freeparams()
        Lbampl = self.get_lb_ampls(obsv)  #LbDecay_lLb_lLc_lw_lwd_WC_FF
        Wampl  = self.get_w_ampls(obsv)   #WDecay_lw_lwd_WC_ll
        #print(len(LbDecay_lLb_lLc_lw_lwd_WC_FF.keys()))    #192 reduces to 92 if Lb is unpolarised
        #print(len(WDecay_lw_lwd_WC_ll.keys()))             #32
        #print(len(fp_wc_ff.keys()))                        #6*20 = 120

        dens = atfi.const(0.)
        for lLb in self.lLbs:
            for lLc in self.lLcs:
                for ll in self.lls:
                    ampls_coherent = atfi.const(0.)
                    for lw in self.lws:
                        for lwd in self.lws:
                            for WC in self.WCs:
                                for FF in self.FFs:
                                    lbindx = (lLb,lLc,lw,lwd,WC,FF)
                                    windx  = (lw,lwd,_replace_wc(WC),ll)
                                    fpindx = (WC, FF)
                                    cond   = lbindx not in Lbampl or windx not in Wampl
                                    if cond: continue
                                    amp  = atfi.sqrt(np.array(2.))*atfi.const(self.GF)*atfi.const(self.Vcb)*atfi.const(self.signWC[WC])*atfi.const(self.eta[lw])*atfi.const(self.eta[lwd])
                                    amp *= Lbampl[lbindx]*Wampl[windx]*free_params[fpindx]
                                    #print(amp)
                                    ampls_coherent+=amp
    
                    dens += atfd.density(ampls_coherent)
    
        return dens

    def get_freeparams_indp_terms(self, ph, getIntegral = False):
        """Get FF and WC independent terms or integrals"""
    
        #define a helpful function
        def _replace_wc(strg):
            """Helpful function when for leptonic amplitudes in definition of density function"""
            strg_wc = strg.replace('A' , 'V')
            strg_wc = strg_wc.replace('PS', 'S')
            strg_wc = strg_wc.replace('PT', 'T')
            return strg_wc
    
        #get Lb and w amplitudes
        obsv       = self.prepare_data(ph) 
        dGdO_phsp  = self.get_phasespace_term(obsv)
        Lbampl     = self.get_lb_ampls(obsv) 
        Wampl      = self.get_w_ampls(obsv)  
        
        fp_indp_terms = {}
        #THIS non-optimal multiple for loop implementation is seems better then tensor contraction in TF. Specially in terms of memory.
        #Also I found it is better than introducing switches as place holder (in TF1) since integration has to be looped and done over each term, whereas a dictionary of integral can be directly evaluated.
        for WC in self.WCs:
            for FF in self.FFs:
                for WC_d in self.WCs:
                    for FF_d in self.FFs:
                        fpindx   = (WC,   FF)
                        fpindx_d = (WC_d, FF_d)
                        if (fpindx_d, fpindx) in fp_indp_terms: continue
                        dens   = atfi.const(0.)
                        filled = False #Only store stuff that gets evaluated (i.e. actually that reaches the end of the loops below pass hurdles
                        for lLb in self.lLbs:
                            for lLc in self.lLcs:
                                for ll in self.lls:
                                    for lw in self.lws:
                                        for lwd in self.lws:
                                            lbindx = (lLb,lLc,lw,lwd,WC,FF)
                                            windx  = (lw,lwd,_replace_wc(WC),ll)
                                            cond1  = lbindx not in Lbampl or windx not in Wampl
                                            if cond1: continue #hurdle1
                                            amp1  = atfi.sqrt(np.array(2.))*atfi.const(self.GF)*atfi.const(self.Vcb)*atfi.const(self.signWC[WC])*atfi.const(self.eta[lw])*atfi.const(self.eta[lwd])
                                            amp1 *= Lbampl[lbindx]*Wampl[windx] #real
                                            for lLb_d in self.lLbs:
                                                for lLc_d in self.lLcs:
                                                    for ll_d in self.lls:
                                                        for lw_d in self.lws:
                                                            for lwd_d in self.lws:
                                                                lbindx_d = (lLb_d,lLc_d,lw_d,lwd_d,WC_d,FF_d)
                                                                windx_d  = (lw_d,lwd_d,_replace_wc(WC_d),ll_d)
                                                                cond2 = lbindx_d not in Lbampl or windx_d not in Wampl
                                                                cond3 = lLb_d != lLb or lLc_d != lLc or ll_d != ll
                                                                if cond2 or cond3: continue #hurdle2
                                                                amp2  = atfi.sqrt(np.array(2.))*atfi.const(self.GF)*atfi.const(self.Vcb)*atfi.const(self.signWC[WC_d])*atfi.const(self.eta[lw_d])*atfi.const(self.eta[lwd_d])
                                                                amp2 *= Lbampl[lbindx_d]*Wampl[windx_d] #real
                                                                filled = True
                                                                if fpindx == fpindx_d: 
                                                                    dens += amp1 * amp2  #2nd terms must be conjugate but here they are both real
                                                                else:
                                                                    dens += atfi.const(2.) * amp1 * amp2 #2nd terms must be conjugate but here they are both real
    
                        if filled: #only store the non-zero or the ones that pass hurdels
                            if getIntegral:
                                fp_indp_terms[(fpindx,fpindx_d)] = (atfi.reduce_mean(dGdO_phsp * dens)).numpy()
                            else:
                                fp_indp_terms[(fpindx,fpindx_d)] = dGdO_phsp * dens 

        return fp_indp_terms
    
    def contract_with_freeparams(self, indp_terms):
        free_params   = self.get_freeparams()
        dynamic_dens  = atfi.sum([free_params[indx[0]] * free_params[indx[1]] * indp_terms[indx] for indx in list(indp_terms.keys())]) #note here both free params are real
        return dynamic_dens

    def get_unbinned_model(self, ph, method= '1'):
        """Define the unbinned model for Lb->Lclnu"""
        model = None
        if method == '1':
            obsv              = self.prepare_data(ph) 
            dGdO_dynm         = self.get_dynamic_term(obsv)
            dGdO_phsp         = self.get_phasespace_term(obsv)
            model             = dGdO_phsp*dGdO_dynm
        elif method == '2':
            fp_indp_terms     = self.get_freeparams_indp_terms(ph) #get first free parameter independent terms
            model             = self.contract_with_freeparams(fp_indp_terms) #contract with the wcs
        else:
            raise Exception("Method not recognised")

        return model

    def get_normalised_pdf_values(self, phsp_array, fp_dict):
        """Get normalised pdf values for a given phase space array and dictionary of parameters values"""

        def _pdf_integral(norm_sample, ranges = self.phase_space.ranges): 
            intg = atfi.reduce_mean(self.get_unbinned_model(norm_sample)) * self.get_phsp_volume(ranges)
            return intg

        #get the previous value of wc
        fp_prev_dict = self.get_params_values(param_list = list(fp_dict.keys()))
        #set a new value for wc
        print('Setting new values')
        self.set_params_values(fp_dict, isfitresult = False)
        #unnormalised pdf
        pdf_unnorm = self.get_unbinned_model(phsp_array)
        #print(pdf_unnorm)
        #get the values of the normalised pdf
        norm_smpl  = self.phase_space.unfiltered_sample(1000000)
        pdf_vals   = pdf_unnorm/_pdf_integral(norm_smpl)
        #print(_pdf_integral(norm_smpl))
        #print(pdf_vals)
        #set the old value of wc back
        print('Setting back the old values')
        self.set_params_values(fp_prev_dict, isfitresult = False)
        return pdf_vals

    def get_freeparams_indp_weights(self, obsv, dens_sm):
        """Function to get NP_PDF/SM_PDF"""
    
        #define a helpful function
        def _replace_wc(strg):
            """Helpful function when for leptonic amplitudes in definition of density function"""
            strg_wc = strg.replace('A' , 'V')
            strg_wc = strg_wc.replace('PS', 'S')
            strg_wc = strg_wc.replace('PT', 'T')
            return strg_wc
    
        #get Lb and w amplitudes
        Lbampl     = self.get_lb_ampls(obsv) 
        Wampl      = self.get_w_ampls(obsv)  
        #print(len(Lbampl.keys()))  #192 reduces to 92 if Lb is unpolarised
        #print(len(Wampl.keys())) #32
        
        Weights = {}
        #THIS non-optimal multiple for loop implementation is seems better then tensor contraction in TF. Specially in terms of memory.
        for WC in self.WCs:
            for FF in self.FFs:
                for WC_d in self.WCs:
                    for FF_d in self.FFs:
                        fpindx   = (WC,   FF)
                        fpindx_d = (WC_d, FF_d)
                        dens     = atfi.const(0.)
                        for lLb in self.lLbs:
                            for lLc in self.lLcs:
                                for ll in self.lls:
                                    for lw in self.lws:
                                        for lwd in self.lws:
                                            lbindx = (lLb,lLc,lw,lwd,WC,FF)
                                            windx  = (lw,lwd,_replace_wc(WC),ll)
                                            cond1  = lbindx not in Lbampl or windx not in Wampl
                                            if cond1: continue
                                            amp1  = atfi.sqrt(np.array(2.))*atfi.const(self.GF)*atfi.const(self.Vcb)*atfi.const(self.signWC[WC])*atfi.const(self.eta[lw])*atfi.const(self.eta[lwd])
                                            amp1 *= Lbampl[lbindx]*Wampl[windx]
                                            for lLb_d in self.lLbs:
                                                for lLc_d in self.lLcs:
                                                    for ll_d in self.lls:
                                                        for lw_d in self.lws:
                                                            for lwd_d in self.lws:
                                                                lbindx_d = (lLb_d,lLc_d,lw_d,lwd_d,WC_d,FF_d)
                                                                windx_d  = (lw_d,lwd_d,_replace_wc(WC_d),ll_d)
                                                                cond2 = lbindx_d not in Lbampl or windx_d not in Wampl
                                                                cond3 = lLb_d != lLb or lLc_d != lLc or ll_d != ll
                                                                if cond2 or cond3: continue
                                                                amp2  = atfi.sqrt(np.array(2.))*atfi.const(self.GF)*self.Vcb*atfi.const(self.signWC[WC_d])*atfi.const(self.eta[lw_d])*atfi.const(self.eta[lwd_d])
                                                                amp2 *= Lbampl[lbindx_d]*Wampl[windx_d]
                                                                dens += amp1 * amp2 #both terms are real here
    
                        Weights[(fpindx+fpindx_d)] = (atfi.where(atfi.equal(dens_sm, 0.), atfi.zeros(dens_sm), tf.divide(dens, dens_sm))).numpy() #Note that the phase space factor dGdO_phsp cancel

        return Weights

    def get_weights_wrt_sm(self, ph, fname = './weights.npy'):
        """Get weights NP_PDF/SM_PDF"""
        #NB: The phase space terms are independent free parameters
        self.set_params_values(self.wc_sm  , isfitresult = False)   #set to SM WC values
        self.set_params_values(self.ff_mean, isfitresult = False) #set to FF mean values
        obsv        = self.prepare_data(ph) 
        dynm_SM     = self.get_dynamic_term(obsv)
        np_indp_dynm= self.get_freeparams_indp_weights(obsv, dynm_SM)
        np.save(fname, np_indp_dynm)
        return np_indp_dynm

    def set_params_values(self, res, isfitresult = True):
        """Set the parameters values to the ones given in the dictionary"""
        for k in list(res.keys()):
            for p in list(self.tot_params.values()):
                if p.name == k:
                    if isfitresult:
                        if p.floating():
                            print('Setting', p.name, 'from ', p.numpy(), 'to ', res[k][0])
                            p.update(res[k][0])
                    else:
                        print('Setting', p.name, 'from ', p.numpy(), 'to ', res[k])
                        p.init_value = res[k]
                        p.update(res[k])

    def get_params_values(self, param_list = None):
        """Get the dictionary of parameters values given a list of all parameters (default returns all parameters)"""
        fp_val_dict = {}
        for p in list(self.tot_params.values()):
            if param_list is None:
                fp_val_dict[p.name] = p.numpy()
            else:
                if p.name in param_list: fp_val_dict[p.name] = p.numpy()

        return fp_val_dict

    def get_phsp_volume(self, lmts):
        """Get the phase space model given the bin limits"""
        return functools.reduce(lambda x, y: x*y, [r[1] - r[0] for r in lmts])

    def get_binned_model(self, bin_scheme = 'Scheme0', applyEff = False, applyResponse = False, 
                         eff_fname = fitdir+'/responsematrix_eff/Eff.p', 
                         res_fname = fitdir+'/responsematrix_eff/responsematrix.p'):
        """Define the binned model using the cached integrals"""
        #Get free parameter independent (nterms x nbins) matrix
        k_fname   = fitdir+'/BinningSchemes/'+bin_scheme+'/keys0.p'
        k_fpdinp  = pickle.load(open(k_fname, 'rb')) #288 keys to map integral value with freeparams (wc,ff)
        print('Number of keys for the free parameter independent integrals', len(k_fpdinp))
        BinScheme = defing_binning_scheme()
        nx        = len(BinScheme[bin_scheme]['qsq'])  - 1
        ny        = len(BinScheme[bin_scheme]['cthl']) - 1
        tot_bins  = nx * ny
        fp_indp_intgl = []
        for binnum in range(tot_bins):
            bin_name = 'Bin'+str(binnum)
            f_name   = fitdir+'/BinningSchemes/'+bin_scheme+'/'+bin_name+'.p'
            bin_lmts = BinScheme[bin_scheme]['bin_limits'][bin_name]
            bin_vol  = self.get_phsp_volume(bin_lmts)
            fp_indp_intgl += [np.array(pickle.load(open(f_name,'rb')), dtype = np.float64) * bin_vol]

        np_fpdinps = np.array(fp_indp_intgl).T #integral values of shape (288, nbins)
        tf_fpdinps = atfi.const(np_fpdinps)
        print('Shape of the free parameter independent integrals (should be 288, nbins)', np_fpdinps.shape)

        #get efficiency map
        efftrue = tf.ones(shape=(nx,ny),dtype=tf.dtypes.float64) #here efficiency is uniform
        if applyEff:
            print('Apply eff') #shape is (nx,ny)
            efftrue     = atfi.const(pickle.load( open( eff_fname, 'rb' ) ))

        #response matrix is identity with ijkl as index where ij (reco q2 and cthl) and kl (true q2 and cthl)
        np_mijkl = np.zeros((nx,ny,nx,ny))
        for i in range(nx):
            for j in range(ny):
                np_mijkl[i][j][i][j] = 1.

        mijkl = atfi.const(np_mijkl) #here reco == true
        if applyResponse:
            print('Apply response matrix on normalised pdf')
            mijkl    = atfi.const(pickle.load( open( res_fname, 'rb' ))) 

        #make a function to be returned
        @atfi.function
        def _binned_model():
            #build the model with free parameters included
            free_params  = self.get_freeparams()
            tf_fps       = tf.reshape(tf.stack([free_params[indx[0]] * free_params[indx[1]] for indx in k_fpdinp]), [1,-1]) #2nd term should be conjugate, but they are both real for us
            Mdl_unnorm   = tf.reshape(tf.einsum('ij,jk->k', tf_fps, tf_fpdinps), (nx,ny)) #i=1,j=terms,k=nbins and reshape to (nx,ny)
            #fold efficiency: before normalising the pdf
            Mdl_unnorm  *= efftrue
            #normalise model
            Mdl_intg     = atfi.reduce_sum(Mdl_unnorm)
            Mdl_norm     = Mdl_unnorm/Mdl_intg
            #fold resolution: after normalising the pdf
            Mdl_norm     = tf.einsum('ijkl,kl->ij', mijkl, Mdl_norm) #should be normalised
            return Mdl_norm
    
        return _binned_model

    def sample_ff_values(self, seed = None, verbose = True):
        """Sample a FF value from multi-dimensional gaussian distribution whose parameters from LQCD"""
        if seed is not None: np.random.seed(seed)

        if (self.ff_floated_mean is None) or (self.ff_floated_cov is None):
            print('FF parameters are all fixed so not randomising them!')
            return None

        ffmeanmod = np.random.multivariate_normal(mean=self.ff_floated_mean, cov=self.ff_floated_cov) 
        if verbose: print('Sampled FFs:', self.ff_floated_names, ',old mean:', self.ff_floated_mean, ', new mean:', ffmeanmod)
        return ffmeanmod

    def randomise_ff_params(self, seed = None, verbose = True):
        """Randomise the FF parameters"""
        if (self.ff_floated_mean is None) or (self.ff_floated_cov is None):
            print('FF parameters are all fixed so not randomising them!')
            return None

        newffvals   = self.sample_ff_values(seed = seed, verbose = False)
        for ffp, ffval in zip(self.ff_floated, newffvals):
            ffprev = ffp.numpy()
            ffp.update(ffval)
            if verbose: print('Randomising FF', ffp.name, 'from', ffprev, 'to', ffp.numpy())

    def randomise_wc_params(self, seed = None, verbose = True):
        """Randomise the wc parameters"""
        if seed is not None: np.random.seed(seed)

        for p in list(self.wc_params.values()):
            if p.floating():
                wcprev = p.numpy()
                newval = np.random.uniform(p.lower_limit, p.upper_limit, size=1)[0]
                p.update(newval)
                if verbose: print('Randomising WC', p.name, 'from', wcprev, 'to', p.numpy())

    def import_unbinned_data(self, fname = './test.root'):
        df = read_root(fname, columns=['q2', 'w_ctheta_l'])
        np_df = df.to_numpy()
        #print(np_df)
        return np_df

    def generate_unbinned_data(self, size, seed = None, chunks = 1000000, store_file = False, fname = './test.root'):
        """Generate unbinned data"""
        if seed is not None:
            atfi.set_seed(seed)
            np.random.seed(seed)

        maximum = tft.maximum_estimator(self.get_unbinned_model, self.phase_space, 100000) * 1.5
        print("Maximum = ", maximum)
        sample  = tft.run_toymc(self.get_unbinned_model, self.phase_space, size, maximum, chunk=chunks)
        print('Sample shape', sample.shape)
        if store_file:
            dt  = self.prepare_data(sample)
            df  = pd.DataFrame.from_dict(dt)
            #print(df)
            to_root(df, fname, key='tree', store_index=False)

        return sample

    def generate_binned_data(self, size, model_binned, seed = None):
        """Generate binned data: note model_binned is callable function"""
        pois   = tfd.Poisson(rate = float(size) * model_binned()) 
        b_data = (pois.sample(1, seed=seed)[0,:]).numpy()
        return b_data

    def generate_binned_data_alternate(self, size, bin_scheme, seed = None, chunks = 1000000, store_file = False, fname = './test.root', import_file = False):
        """Generate binned data, two methods available: binned_1 (from unbinned to binned) and binned_2 (binned directly)"""

        if import_file:
            sample = self.import_unbinned_data(fname = fname)
        else:
            sample = self.generate_unbinned_data(size, seed = seed, chunks = chunks, store_file = store_file, fname = fname)
            
        BinScheme = defing_binning_scheme()
        xedges = BinScheme[bin_scheme]['qsq']
        yedges = BinScheme[bin_scheme]['cthl']
        b_data = np.float64(np.histogram2d(sample[:,0], sample[:,1], bins=(xedges, yedges), range=self.phase_space.ranges)[0]) 
        return b_data

    def gaussian_constraint_ffparams(self): 
        """Gaussian constrain the form factor parameters that are floated"""
        loglh_func = None
        if (self.ff_floated_mean is None) or (self.ff_floated_cov is None):
            loglh_func  = lambda : atfi.const(0.)
            print("All FF params are fixed so returning zero for log likelihood of gaussian constriant!")
        else:
            ffmodmean     = self.sample_ff_values(verbose = True)
            multigaus     = tfd.MultivariateNormalTriL(loc=ffmodmean, scale_tril=tf.linalg.cholesky(self.ff_floated_cov))
            loglh_func    = lambda : multigaus.log_prob([p() for p in self.ff_floated])

        return loglh_func

    def binned_nll(self, data_binned, model_binned, gauss_constraint_ff = False): 
        #data_binned is unnormalised and model_binned is normalised
        n_tot = atfi.reduce_sum(data_binned)
        #The mean of the gaussian is sampled once and should NOT change during each minimisation step (hence outside of _nll function)
        #However the value of log likelihood MUST change depending on the parameter value at each minimisation step (therefore a lambda function)
        nll_gaus_func = lambda: atfi.const(0.)
        if gauss_constraint_ff:
            nll_gaus_func =  self.gaussian_constraint_ffparams()  #Ltot = L * Lg => NLLtot = NLL - LogLg

        @atfi.function
        def _nll(pars): 
            mu_i  = n_tot * model_binned()
            mu_tot= atfi.reduce_sum(mu_i) #equal to n_tot here
            NLL   = mu_tot - n_tot - atfi.reduce_sum(data_binned * atfi.log(mu_i))
            NLL  -= nll_gaus_func()
            return NLL

        return _nll

    def plot_fitresult(self, np_data, np_fit, bin_scheme = 'Scheme0', fname = './fitres.pdf', xlabel = "q^{2} [GeV^{2}]", ylabel = "cos(#theta_{#mu})"):
        """plot the results of fit along with pull"""
        np_data =  np_data.flatten()
        np_fit  =  np_fit.flatten()
        #print(np_data.shape)
        #print(np_fit.shape)
    
        #print(np_fit.sum())
        np_fit = np_data.sum()/np_fit.sum() * np_fit #np_data unnormalised

        from ROOT import TFile, TTree, TH1D, TH2D, gROOT, gStyle, TCanvas, kGreen, kRed, TCut, TPaveText, TLegend, kBlue, gPad
        gROOT.SetBatch(True)
        gStyle.SetOptStat(0);  
        gStyle.SetOptTitle(0);
        gROOT.ProcessLine(".x plots/lhcbStyle2D.C")
    
        BinScheme = defing_binning_scheme()
        xedges    = BinScheme[bin_scheme]['qsq']
        yedges    = BinScheme[bin_scheme]['cthl']
        bcntrs    = BinScheme[bin_scheme]['bin_centers']
        nx        = len(BinScheme[bin_scheme]['qsq'])  - 1
        ny        = len(BinScheme[bin_scheme]['cthl']) - 1
        tot_bins  = nx * ny
    
        hdata = TH2D("hdata", "hdata", nx, xedges, ny, yedges);
        hdata.SetXTitle(xlabel); hdata.SetYTitle(ylabel); hdata.SetTitle("Data"); hdata.SetZTitle("Events/Bin")
        hfit  = TH2D("hfit", "hfit", nx, xedges, ny, yedges);
        hfit.SetXTitle(xlabel); hfit.SetYTitle(ylabel); hfit.SetTitle("Fit"); hfit.SetZTitle("Events/Bin")
        hpull = TH2D("hpull", "hpull", nx, xedges, ny, yedges);
        hpull.SetXTitle(xlabel); hpull.SetYTitle(ylabel); hpull.SetTitle("Pull"); hpull.SetZTitle("Pull")

        for i in range(tot_bins):
            bin_name = 'Bin'+str(i)
            bcntr    = BinScheme[bin_scheme]['bin_centers'][bin_name]
            #print(bcntr)
            glbbin_d = hdata.FindBin(bcntr[0], bcntr[1])
            glbbin_f =  hfit.FindBin(bcntr[0], bcntr[1])
            glbbin_p = hpull.FindBin(bcntr[0], bcntr[1])
            hdata.SetBinContent(glbbin_d, np_data[i])
            hfit.SetBinContent(glbbin_f,   np_fit[i])
            pull_nm = (np_data[i] - np_fit[i])
            pull_dn = np.sqrt(np_data[i] + np_fit[i])
            if pull_dn == 0.:
                hpull.SetBinContent(glbbin_p,  0.)
            else:
                hpull.SetBinContent(glbbin_p,  pull_nm/pull_dn)
    
        c1 = TCanvas("c1","c1")
        hdata.Draw("colz")
        c1.SaveAs(fname.replace('.pdf', '_data.pdf'))
    
        c2 = TCanvas("c2","c2")
        hfit.Draw("colz")
        c2.SaveAs(fname.replace('.pdf', '_fit.pdf'))
    
        c3 = TCanvas("c3","c3")
        gStyle.SetPalette(70)
        hpull.Draw("colz")
        c3.SaveAs(fname.replace('.pdf', '_pull.pdf'))

    def write_fit_results(self, results, filename):
        f = open(filename, "w")
        floated_params = None
        if (self.ff_floated is None) and (self.wc_floated is not None):
            floated_params = self.wc_floated
        elif (self.ff_floated is not None) and (self.wc_floated is None):
            floated_params = self.ff_floated
        elif (self.ff_floated is not None) and (self.wc_floated is not None):
            floated_params = self.ff_floated + self.wc_floated
        else:
            raise Exception("Nothing is floated, please check!")

        for p in floated_params:
            s = "%s " % p.name
            for i in results["params"][p.name]:
                s += "%f " % i

            f.write(s + "\n")

        s = "loglh %f\n"                   % (results["loglh"])
        #see here to understand what below mean: https://iminuit.readthedocs.io/en/stable/reference.html
        #is_valid == (has_valid_parameters & !has_reached_call_limit & !is_above_max_edm)
        s+= "is_valid %i\n"                % (results["is_valid"])
        s+= "has_parameters_at_limit %i\n" % (results["has_parameters_at_limit"])
        s+= "has_accurate_covar %i\n"      % (results["has_accurate_covar"])
        s+= "has_posdef_covar %i\n"        % (results["has_posdef_covar"])
        s+= "has_made_posdef_covar %i"   % (results["has_made_posdef_covar"])
        f.write(s + "\n")
        f.close()

        f = open(filename.replace('.txt', '_covmatrix.txt'), "w")
        for k1 in list(results["covmatrix"].keys()):
            for k2 in list(results["covmatrix"].keys()):
                s = '{0} {1} {2}'.format(k1, k2, results["covmatrix"][k1][k2])
                f.write(s + "\n")
        f.close()

########### Define other useful functions below
def Minimize(nll, model, tot_params, nfits = 1, use_hesse = True, use_minos = False, use_grad = False, randomiseFF = True):
    nllval = None
    reslts = None
    for nfit in range(nfits):
        #Randomising the starting values of wc parameters according a uniform distribution b/w [-2., 2.]. 
        #NB: This is also the allowed range in which the fit can vary these parameters when minimising nll.
        model.randomise_wc_params()

        #Randomising the starting values of ff parameters according a multidimensional gaussian distibution from Lattice QCD (LQCD) paper.
        #NB1: The allowed ranges for the form factors is  [ff_mean - 20. * ff_sigma, ff_mean + 20. * ff_sigma] where ff_mean and ff_sigma is the central value and uncertainty as measured in LQCD paper.
        #NB2: If FF are not floated then they are not randomised
        if randomiseFF:
            model.randomise_ff_params()

        #Conduct the fit
        results = tfo.run_minuit(nll, list(tot_params.values()), use_gradient=use_grad, use_hesse = use_hesse, use_minos = use_minos, get_covariance = True)

        #out of nfits pick the result with the least negative log likelihood (NLL)
        if nfit == 0: 
            print('Fit number', nfit)
            nllval = results['loglh']
            reslts = results
        else:
            print('Fit number', nfit)
            if nllval > results['loglh']:
                nllval = results['loglh']
                reslts = results
    
    #set the parameters to the fit results of the least NLL
    model.set_params_values(reslts['params'], isfitresult = True)
    return reslts

def str2bool(v):
    """Function used in argparse"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def MakeFourVector(pmag, costheta, phi, msq): 
    """Make 4-vec given magnitude, cos(theta), phi and mass"""
    sintheta = atfi.sqrt(1. - costheta**2) #theta 0 to pi => atfi.sin always pos
    px = pmag * sintheta * atfi.cos(phi)
    py = pmag * sintheta * atfi.sin(phi)
    pz = pmag * costheta
    E  = atfi.sqrt(msq + pmag**2)
    return atfk.lorentz_vector(atfk.vector(px, py, pz), E)

def pvecMag(Msq, m1sq, m2sq): 
    """Momentum mag of (1 or 2) in M rest frame"""
    kallen = atfi.sqrt(Msq**2 + m1sq**2 + m2sq**2 - 2.*(Msq*m1sq + Msq*m2sq + m1sq*m2sq))
    return kallen/(2.*atfi.sqrt(Msq))

def InvRotateAndBoostFromRest(m_p4mom_p1, m_p4mom_p2, gm_phi_m, gm_ctheta_m, gm_p4mom_m):
    """
    Function that gives 4momenta of the particles in the grand mother helicity frame

    m_p4mom_p1: particle p1 4mom in mother m helicity frame (i.e. z points along m momentum in grand mother gm's rest frame)
    m_p4mom_p2: particle p2 4mom in mother m helicity frame (i.e. z points along m momentum in grand mother gm's rest frame)
    gm_phi_m, gm_ctheta_m: Angles phi and ctheta of m in gm's helicity frame (i.e z points along gm momentum in grand-grand-mother rest frame)
    gm_4mom_m: m 4mom in gm's helicity frame
    --------------
    Checks done such as 
        - (lc_p4mom_p+lc_p4mom_k == lc_p4mom_r), 
        - Going in reverse i.e. boost lc_p4mom_p into R rest frame (using lc_p4mom_r) and rotating into R helicity frame gives back r_p4mom_p 
            - i.e. lc_p4mom_p_opp = atfk.rotate_lorentz_vector(atfk.boost_to_rest(lc_p4mom_p, lc_p4mom_r), -lc_phi_r, -atfi.acos(lc_ctheta_r), lc_phi_r)) where lc_p4mom_p_opp == r_p4mom_p
        - Rotate and boost, instead of boost and rotate should give the same answer, checked (Does not agree with TFA implementation though)
            -i.e. lc_p4mom_prot  = atfk.rotate_lorentz_vector(lc_p4mom_p, -lc_phi_r, -atfi.acos(lc_ctheta_r), lc_phi_r)
            -i.e. lc_p4mom_krot  = atfk.rotate_lorentz_vector(lc_p4mom_k, -lc_phi_r, -atfi.acos(lc_ctheta_r), lc_phi_r)
            -i.e. lc_p4mom_p2    = atfk.boost_to_rest(lc_p4mom_prot, lc_p4mom_prot+lc_p4mom_krot) #lc_p4mom_p2 == r_p4mom_p
            -i.e. lc_p4mom_k2    = atfk.boost_to_rest(lc_p4mom_krot, lc_p4mom_prot+lc_p4mom_krot) #lc_p4mom_k2 == r_p4mom_k
    """
    #First: Rotate particle p 3mom defined in mother m helicity frame such that z now points along z in grand mother gm helicity frame 
    #(i.e interpreted as gm mom in it's grand-grand-mother ggm rest frame)
    m_p4mom_p1_zgmmom      = atfk.rotate_lorentz_vector(m_p4mom_p1, -gm_phi_m, atfi.acos(gm_ctheta_m), gm_phi_m)
    m_p4mom_p2_zgmmom      = atfk.rotate_lorentz_vector(m_p4mom_p2, -gm_phi_m, atfi.acos(gm_ctheta_m), gm_phi_m)
    #Second: Boost particle p 4mom from mother m's rest frame to gm helicity frame. This is done using boost vector from gm_p4mom_m  [checked: E(m_p4mom_p1_zmmom + m_p4mom_p2_zmmom) == Mass_m]
    gm_p4mom_p1            = atfk.boost_from_rest(m_p4mom_p1_zgmmom, gm_p4mom_m)
    gm_p4mom_p2            = atfk.boost_from_rest(m_p4mom_p2_zgmmom, gm_p4mom_m)
    return gm_p4mom_p1, gm_p4mom_p2

def make_q2_cthl(dt_np, Lb4moml, Lc4moml, Mu4moml, Nu4moml):
    """Get q^2 and cos(theta) given 4 momenta of all particles"""
    Lb4mom = atfk.lorentz_vector(atfk.vector(dt_np[Lb4moml[1]],dt_np[Lb4moml[2]],dt_np[Lb4moml[3]]), dt_np[Lb4moml[0]])
    Lc4mom = atfk.lorentz_vector(atfk.vector(dt_np[Lc4moml[1]],dt_np[Lc4moml[2]],dt_np[Lc4moml[3]]), dt_np[Lc4moml[0]])
    Mu4mom = atfk.lorentz_vector(atfk.vector(dt_np[Mu4moml[1]],dt_np[Mu4moml[2]],dt_np[Mu4moml[3]]), dt_np[Mu4moml[0]])
    Nu4mom = atfk.lorentz_vector(atfk.vector(dt_np[Nu4moml[1]],dt_np[Nu4moml[2]],dt_np[Nu4moml[3]]), dt_np[Nu4moml[0]])
    #Nu4mom = Nu4mom + (Lb4mom-(Lc4mom+Mu4mom+Nu4mom))
    #Boost Lb frame
    Lb4mom_Lb  = atfk.boost_to_rest(Lb4mom, Lb4mom)
    Lc4mom_Lb  = atfk.boost_to_rest(Lc4mom, Lb4mom)
    Mu4mom_Lb  = atfk.boost_to_rest(Mu4mom, Lb4mom)
    Nu4mom_Lb  = atfk.boost_to_rest(Nu4mom, Lb4mom)
    W4mom_Lb   = atfk.boost_to_rest(Mu4mom+Nu4mom, Lb4mom)
    #Q2
    Q2_var     = tf.square(atfk.mass(Mu4mom+Nu4mom))
    #Cthl
    Lb4mom_W  = atfk.boost_to_rest(Lb4mom_Lb, W4mom_Lb)
    Mu4mom_W  = atfk.boost_to_rest(Mu4mom_Lb, W4mom_Lb)
    Cthl_var  =-atfk.scalar_product(atfk.spatial_components(Mu4mom_W), atfk.spatial_components(Lb4mom_W))/(P(Mu4mom_W) * P(Lb4mom_W))
    #clthl    = atfk.scalar_product(atfk.spatial_components(Mu4mom_W), atfk.spatial_components(W4mom_Lb))/(P(Mu4mom_W) * P(W4mom_Lb))
    return Q2_var, Cthl_var

def HelAngles3Body(pa, pb, pc):
    """Get three-body helicity angles"""
    theta_r  = atfi.acos(-atfk.z_component(pc) / atfk.norm(atfk.spatial_components(pc)))
    phi_r    = atfi.atan2(-atfk.y_component(pc), -atfk.x_component(pc))
    pa_prime = atfk.rotate_lorentz_vector(pa, -phi_r, -theta_r, phi_r)
    pb_prime = atfk.rotate_lorentz_vector(pb, -phi_r, -theta_r, phi_r)
    pa_prime2= atfk.boost_to_rest(pa_prime, pa_prime+pb_prime)
    theta_a  = atfi.acos(atfk.z_component(pa_prime2) / atfk.norm(atfk.spatial_components(pa_prime2)))
    phi_a    = atfi.atan2(atfk.y_component(pa_prime2), atfk.x_component(pa_prime2))
    return (theta_r, phi_r, theta_a, phi_a)

def RotateAndBoostToRest(gm_p4mom_p1, gm_p4mom_p2, gm_phi_m, gm_ctheta_m):
    """
    gm_p4mom_p1: particle p1 4mom in grand mother gm helicity frame (i.e z points along gm momentum in grand-grand-mother rest frame).
    gm_p4mom_p2: particle p2 4mom in grand mother gm helicity frame (i.e z points along gm momentum in grand-grand-mother rest frame).
    gm_phi_m, gm_ctheta_m: Angles phi and ctheta of m in gm's helicity frame (i.e z points along gm momentum in grand-grand-mother rest frame)
    """
    #First:  Rotate particle p 3mom defined in grand mother gm helicity frame such that z now points along z of mother m in gm rest frame.
    gm_p4mom_p1_zmmom      = atfk.rotate_lorentz_vector(gm_p4mom_p1, -gm_phi_m, -atfi.acos(gm_ctheta_m), gm_phi_m)
    gm_p4mom_p2_zmmom      = atfk.rotate_lorentz_vector(gm_p4mom_p2, -gm_phi_m, -atfi.acos(gm_ctheta_m), gm_phi_m)
    #Second: Boost particle p 4mom to mother m i.e. p1p2 rest frame
    gm_p4mom_m             = gm_p4mom_p1_zmmom+gm_p4mom_p2_zmmom
    m_p4mom_p1             = atfk.boost_to_rest(gm_p4mom_p1_zmmom, gm_p4mom_m)
    m_p4mom_p2             = atfk.boost_to_rest(gm_p4mom_p2_zmmom, gm_p4mom_m)
    return m_p4mom_p1, m_p4mom_p2

class Old_Model():
    """Implementations of the old chinese model! Not used in this ANALYSIS, the normalised values of these match the model we use!"""
    def __init__(self, MLb, MLc, Mlep):
        self.MLb     = atfi.const(MLb)
        self.MLc     = atfi.const(MLc)
        self.Mlep    = atfi.const(Mlep)
        self.GF      = atfi.const(1.166378e-5) #GeV^-2
        self.Vcb     = atfi.const(4.22e-2)     #avg of incl and excl
        self.mb      = atfi.const(4.18       ) #GeV
        self.mc      = atfi.const(1.275      ) #GeV
        self.Mpos    = atfi.const(self.MLb) + atfi.const(self.MLc)
        self.Mneg    = atfi.const(self.MLb) - atfi.const(self.MLc)
        self.t0      = atfi.pow(self.Mneg,2)

    def _unnormalized_pdf(self, x, params):
        #observables
        q2      = x[:,0]
        costhl  = x[:,1]
        #Wilson coefficients 
        CVR = params['CVR']()
        CVL = params['CVL']()
        CSR = params['CSR']()
        CSL = params['CSL']()
        CT  = params['CT' ]()
        #Form factors
        FV1, FV2, FV3, FA1, FA2, FA3, fT, gT, fVT, gVT, fST, gST = self.GetFormFactors(q2, params)
        #Preliminary
        Q_p     = self.Mpos**2 - q2 
        Q_n     = self.Mneg**2 - q2 
        mag_p2  = atfi.sqrt(Q_p * Q_n)/2./self.MLb
        #Helicity amplitudes: 
        #Note in the new paper: (Vector) f1 = FV1, f2 = -FV2/self.MLb and f3 = FV3/self.MLb and (Axial) same for g2, g2, g3
        #HV: 
        HV_ph_pt = (1. + CVL + CVR) * atfi.sqrt(Q_p)/atfi.sqrt(q2) * (self.Mneg * FV1 + q2/self.MLb * FV3)
        HV_ph_p  = (1. + CVL + CVR) * atfi.sqrt(2 * Q_n) * (-FV1 - self.Mpos/self.MLb * FV2) #use the one in the new paper
        HV_ph_z  = (1. + CVL + CVR) * atfi.sqrt(Q_n)/atfi.sqrt(q2) * (self.Mpos * FV1 + q2/self.MLb * FV2)
        HV_nh_nt = HV_ph_pt
        HV_nh_n  = HV_ph_p
        HV_nh_z  = HV_ph_z
        #HA
        HA_ph_pt = (1. + CVL - CVR) * atfi.sqrt(Q_n)/atfi.sqrt(q2) * (self.Mpos * FA1 - q2/self.MLb * FA3)
        HA_ph_p  = (1. + CVL - CVR) * atfi.sqrt(2 * Q_p) * (-FA1 + self.Mneg/self.MLb * FA2) #use the one in the new paper
        HA_ph_z  = (1. + CVL - CVR) * atfi.sqrt(Q_p)/atfi.sqrt(q2) * (self.Mneg * FA1 - q2/self.MLb * FA2)
        HA_nh_nt = - HA_ph_pt
        HA_nh_n  = - HA_ph_p
        HA_nh_z  = - HA_ph_z
        #H: HV - HA
        H_ph_pt = HV_ph_pt   -  HA_ph_pt
        H_ph_p  = HV_ph_p    -  HA_ph_p 
        H_ph_z  = HV_ph_z    -  HA_ph_z 
        H_nh_nt = HV_nh_nt   -  HA_nh_nt
        H_nh_n  = HV_nh_n    -  HA_nh_n 
        H_nh_z  = HV_nh_z    -  HA_nh_z 
        H_nh_pt = H_nh_nt #since t = 0 with JW = 0
        #HSP_x_y: Scalar - PsuedoScalar
        HSP_ph_z = (CSL + CSR) * atfi.sqrt(Q_p)/(self.mb - self.mc) * (FV1 * self.Mneg + FV3 * q2/self.MLb) + (CSL - CSR) * atfi.sqrt(Q_n)/(self.mb + self.mc) * (FA1 * self.Mpos - FA3 * q2/self.MLb)
        HSP_nh_z = (CSL + CSR) * atfi.sqrt(Q_p)/(self.mb - self.mc) * (FV1 * self.Mneg + FV3 * q2/self.MLb) - (CSL - CSR) * atfi.sqrt(Q_n)/(self.mb + self.mc) * (FA1 * self.Mpos - FA3 * q2/self.MLb)
        #HT_x_y_z: Tensor
        HT_ph_p_z   = -CT * atfi.sqrt(2./q2) * (fT * atfi.sqrt(Q_p) * self.Mneg + gT * atfi.sqrt(Q_n) * self.Mpos)
        HT_ph_p_n   = -CT * (fT * atfi.sqrt(Q_p) + gT * atfi.sqrt(Q_n))
        HT_ph_p_pt  =  CT * (- atfi.sqrt(2./q2) * (fT * atfi.sqrt(Q_n) * self.Mpos + gT * atfi.sqrt(Q_p) * self.Mneg) +  atfi.sqrt(2.*q2) * (fVT * atfi.sqrt(Q_n) - gVT * atfi.sqrt(Q_p)))
        HT_ph_z_pt  =  CT * (- fT * atfi.sqrt(Q_n) - gT * atfi.sqrt(Q_p) + fVT * atfi.sqrt(Q_n) * self.Mpos - gVT * atfi.sqrt(Q_p) * self.Mneg + fST * atfi.sqrt(Q_n) * Q_p + gST * atfi.sqrt(Q_p) * Q_n )
        HT_nh_p_n   =  CT * (fT * atfi.sqrt(Q_p) - gT * atfi.sqrt(Q_n))
        HT_nh_z_n   =  CT * (atfi.sqrt(2./q2) * (fT * atfi.sqrt(Q_p) * self.Mneg - gT * atfi.sqrt(Q_n) * self.Mpos))
        HT_nh_z_pt  =  CT * (-fT * atfi.sqrt(Q_n) + gT * atfi.sqrt(Q_p) + fVT * atfi.sqrt(Q_n) * self.Mpos + gVT * atfi.sqrt(Q_p) * self.Mneg + fST * atfi.sqrt(Q_n) * Q_p - gST * atfi.sqrt(Q_p) * Q_n)
        HT_nh_n_pt  =  CT * (-atfi.sqrt(2./q2) * (fT * atfi.sqrt(Q_n) * self.Mpos - gT * atfi.sqrt(Q_p) * self.Mneg) + atfi.sqrt(2. * q2) * (fVT * atfi.sqrt(Q_n) + gVT * atfi.sqrt(Q_p)))
        ##Relation: HT_L_l1_l2 =  - HT_L_l2_l1
        #HT_ph_z_p   = - HT_ph_p_z
        #HT_ph_n_p   = - HT_ph_p_n
        #HT_ph_pt_p  = - HT_ph_p_pt
        #HT_ph_pt_z  = - HT_ph_z_pt
        #HT_nh_n_p   = - HT_nh_p_n
        #HT_nh_n_z   = - HT_nh_z_n
        #HT_nh_pt_z  = - HT_nh_z_pt
        #HT_nh_pt_n  = - HT_nh_n_pt
        #N
        N = (self.GF**2 * self.Vcb**2 * q2 * mag_p2)/(512. * atfi.pi()**3. * self.MLb**2) * (1 - self.Mlep**2/q2)**2
        #A1
        A1 = (2. * (1 - costhl**2) * (H_ph_z**2 + H_nh_z**2) + (1 - costhl)**2 * H_ph_p**2 + (1 + costhl)**2 * H_nh_n**2 )
        #AV2
        AV2  = (2. * costhl**2 * (H_ph_z**2 + H_nh_z**2) + (1 - costhl**2) * (H_ph_p**2 + H_nh_n**2) + 2 * (H_ph_pt**2 + H_nh_pt**2)) 
        AV2 -= (4 * costhl * (H_ph_z * H_ph_pt + H_nh_z * H_nh_pt))
        #AT2
        AT2  = 1./4. * (2. * (1 - costhl**2) * (HT_ph_p_n**2 + HT_ph_z_pt**2 + HT_nh_p_n**2 + HT_nh_z_pt**2 + 2. * HT_ph_p_n * HT_ph_z_pt + 2. * HT_nh_p_n * HT_nh_z_pt))
        AT2 += 1./4. * ((1 + costhl)**2 * (HT_nh_z_n**2 + HT_nh_n_pt**2 + 2. * HT_nh_z_n * HT_nh_n_pt)) 
        AT2 += 1./4. * ((1 - costhl)**2 * (HT_ph_p_z**2 + HT_ph_p_pt**2 + 2. * HT_ph_p_z * HT_ph_p_pt)) 
        #A3
        A3  = 1./8. * (2. * costhl**2 * (HT_ph_p_n**2 + HT_ph_z_pt**2 + HT_nh_p_n**2 + HT_nh_z_pt**2 + 2. * HT_ph_p_n * HT_ph_z_pt + 2. * HT_nh_p_n * HT_nh_z_pt))
        A3 += 1./8. * ((1 - costhl**2) * (HT_ph_p_z**2 + HT_ph_p_pt**2 + HT_nh_z_n**2 + HT_nh_n_pt**2 + 2. * HT_ph_p_z * HT_ph_p_pt + 2. * HT_nh_z_n * HT_nh_n_pt))
        A3 += (HSP_ph_z**2 + HSP_nh_z**2)
        #A4
        A4  = (-costhl * (H_ph_z * HSP_ph_z + H_nh_z * HSP_nh_z) + (H_ph_pt * HSP_ph_z + H_nh_pt * HSP_nh_z))
        A4 += (costhl**2/2. * (H_ph_z * HT_ph_p_n + H_ph_z * HT_ph_z_pt + H_nh_z * HT_nh_p_n + H_nh_z * HT_nh_z_pt))
        A4 -= (costhl/2. * (H_ph_pt * HT_ph_p_n + H_ph_pt * HT_ph_z_pt + H_nh_pt * HT_nh_p_n + H_nh_pt * HT_nh_z_pt)) 
        A4 += ((1. - costhl)**2/4. * (H_ph_p * HT_ph_p_z + H_ph_p * HT_ph_p_pt))
        A4 += ((1. + costhl)**2/4. * (H_nh_n * HT_nh_z_n + H_nh_n * HT_nh_n_pt))
        A4 += ((1. - costhl**2)/4. * (H_ph_p * HT_ph_p_z + H_ph_p * HT_ph_p_pt + H_nh_n * HT_nh_z_n + H_nh_n * HT_nh_n_pt + 2. * H_ph_z * HT_ph_p_n + 2. * H_ph_z * HT_ph_z_pt + 2. * H_nh_z * HT_nh_p_n + 2. * H_nh_z * HT_nh_z_pt))
        #A5
        A5 = -2. * costhl * (HSP_ph_z * HT_ph_p_n + HSP_ph_z * HT_ph_z_pt + HSP_nh_z * HT_nh_p_n + HSP_nh_z * HT_nh_z_pt)
        #diff density
        dG_dq2_dcosthl =  N * (A1 + self.Mlep**2/q2 * (AV2 + AT2) + 2. * A3 + 4. * self.Mlep/atfi.sqrt(q2) * A4 + A5)
        return dG_dq2_dcosthl

    @atfi.function
    def GetFormFactors(self, q2, params):
        #define delataf
        Deltaf = {}
        Deltaf['fplus']  = atfi.const(56e-3)  #GeV
        Deltaf['fperp']  = atfi.const(56e-3)  #GeV
        Deltaf['f0']     = atfi.const(449e-3) #GeV
        Deltaf['gplus']  = atfi.const(492e-3) #GeV
        Deltaf['gperp']  = atfi.const(492e-3) #GeV
        Deltaf['g0']     = atfi.const(0.)     #GeV
        M_Bc = tf.constant(6.276, dtype = tf.float64) #GeV, from straub paper
        #define mf_pole, tf_p, zf, Weinberg form factor functions
        mf_pole = lambda deltaf: M_Bc + deltaf 
        tf_plus = lambda Mf_pole: Mf_pole**2
        zf = lambda Q2, Tf_p: (atfi.sqrt(Tf_p - Q2) - atfi.sqrt(Tf_p - self.t0))/(atfi.sqrt(Tf_p - Q2) + atfi.sqrt(Tf_p - self.t0))
        ff = lambda Q2, Mf_pole, A0, A1: 1./(1. - Q2/Mf_pole**2) * (A0 + A1 * zf(Q2, tf_plus(Mf_pole)))
        #Weinber FF
        fplus = ff(q2, mf_pole(Deltaf['fplus']), params['a0fplus'](), params['a1fplus']())
        fperp = ff(q2, mf_pole(Deltaf['fperp']), params['a0fperp'](), params['a1fperp']())
        f0    = ff(q2, mf_pole(Deltaf['f0']),    params['a0f0']   (), params['a1f0'   ]())
        gplus = ff(q2, mf_pole(Deltaf['gplus']), params['a0gplus'](), params['a1gplus']())
        gperp = ff(q2, mf_pole(Deltaf['gperp']), params['a0gplus'](), params['a1gperp']()) #a0gperp == a0gplus
        g0    = ff(q2, mf_pole(Deltaf['g0']),    params['a0g0']   (), params['a1g0'   ]())
        #Do the transformation Weinber -> Helicity ff
        #functions to transform Weinberg ff -> helicity ff
        f2 = lambda Fplus, Fperp, A, B: (Fplus - Fperp)/(A - B)
        f1 = lambda Fperp, B, F2: Fperp - B * F2
        f3 = lambda F0, F1, C: (F0 - F1)/C
        #Helicity ff
        af  = q2/self.MLb/self.Mpos
        bf  = self.Mpos/self.MLb
        cf  = q2/self.MLb/self.Mneg
        ag  = -q2/self.MLb/self.Mneg
        bg  = -self.Mneg/self.MLb
        cg  = -q2/self.MLb/self.Mpos
        FV2 = f2(fplus, fperp, af, bf)
        FV1 = f1(fperp, bf, FV2)
        FV3 = f3(f0, FV1, cf)
        FA2 = f2(gplus, gperp, ag, bg)
        FA1 = f1(gperp, bg, FA2)
        FA3 = f3(g0, FA1, cg)
        #HQET relations for (pseudo-)tensor form factors
        fT  = FV1
        gT  = FV1
        fVT = 0.
        gVT = fVT
        fST = fVT
        gST = fVT
        return (FV1, FV2, FV3, FA1, FA2, FA3, fT, gT, fVT, gVT, fST, gST)
########### 

###########  tests
#MLb     = 5619.49997776e-3    #GeV
#MLc     = 2286.45992749e-3    #GeV
#Mlep    = 105.6583712e-3      #GeV Mu
##Mlep   = 1.77686             #GeV Tau
#
#md       = LbToLclNu_Model(MLb, MLc, Mlep)
#phsp_arr = np.array([[0.1, 0.2], 
#                     [1.3, 0.1],
#                     [5.3,-0.1]])
#
#pdfa = md.get_unbinned_model(phsp_arr, method= '1')
#print(pdfa)
#np_pdfa = pdfa.numpy()
#print(np_pdfa/np_pdfa.sum())
#pdfb = md.get_unbinned_model(phsp_arr, method= '2')

#md.get_unbinned_model(phsp_arr, method= '1')
#md.wc_params['CVR'].update(0.5)
#md.get_unbinned_model(phsp_arr, method= '1')

#md.get_weights_wrt_sm(phsp_arr, fname = './weights.npy')

#md.get_normalised_pdf_values(phsp_arr, {'CVR': 1.0})

#making a callable function is better in tf2 compared to md.generate_binned_data which isnt below
#bmd = md.get_binned_model()
#print(bmd())
#md.tot_params['a0fplus'].update(0.6)
#print(bmd())

#np_data_binned = md.generate_binned_data(int(7.5e6), bmd)
#print(np_data_binned)
#md.tot_params['a0fplus'].update(0.6)
#np_data_binned = md.generate_binned_data(int(7.5e6), bmd)
#print(np_data_binned)

#md.randomise_ff_params()
#md.randomise_wc_params()

#md.generate_unbinned_data(100000, store_file = True, fname = './test.root')
#md.import_unbinned_data(fname = './test.root')

#md.generate_binned_data_alternate(1000, 'Scheme0')

#nl = md.gaussian_constraint_ffparams()
#print(nl)
#md.ff_params['a0fplus'].update(0.6)
#nl = md.gaussian_constraint_ffparams()
#print(nl)

#nl = md.binned_nll(np_data_binned, bmd, gauss_constraint = True)
#print(nl(md.tot_params))
#md.tot_params['a0fplus'].update(0.6)
#print(nl(md.tot_params))

#parms = md.tot_params
#oldmd = Old_Model(MLb, MLc, Mlep)
#pdfb  = oldmd._unnormalized_pdf(phsp_arr, parms)
#print(pdfb)
#np_pdfb = pdfb.numpy()
#print(np_pdfb/np_pdfb.sum())
########### 
