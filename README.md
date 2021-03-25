# About

Repository created for Bachelor's project at Zurich. 
This project is ported from gitlab https://gitlab.cern.ch/amathad/toyangular. 
It contains code to run toy fits to the phase space of $`\Lambda_b \rightarrow \Lambda_c \mu \nu`$ decays to optimise the binning scheme.

# Model

The mathematical expression for the probability density function of our model is given by 

```math
PDF(\vec{x}_R;\vec{\theta}) = \frac{1}{N} \times \int{R(\vec{x}_R,\vec{x}_T) \epsilon(\vec{x}_T) f(\vec{x}_T;\vec{\theta})\,d{\vec{x}_T}}
```

where

- $`\vec{x}_R = (q_R^2, \cos((\theta_\mu))_R)`$ are the reconstruced (R) phase space observables.
- $`\vec{x}_T = (q_T^2, \cos((\theta_\mu))_T)`$ are the true (T) phase space observables.
- $`\vec{\theta}`$ are the parameters of our fit. So wilson coefficients and form factor parameters in our case.
- $`f(\vec{x}_T;\vec{\theta})`$ is the dynamic function that contains the physics. 
- $`\epsilon(\vec{x}_T)`$ is the probability that our signal decays would pass a given selection requirements. Typically in analyses, selection requirements are designed to suppress background. 
- $`R(\vec{x}_R,\vec{x}_T)`$ is the resolution function that encodes the information of the resolution on the reconstructed observables. 
- $`N`$ is the normalisation such that $`\int{PDF(\vec{x}_R;\vec{\theta})\,d{\vec{x}_R}} = 1`$

Since we are doing the binned fit the above equation becomes

```math
PDF(\vec{x}^i_R;\vec{\theta}) = \frac{1}{N} \times \sum_{ij} {R(\vec{x}^i_R,\vec{x}^j_T) \epsilon(\vec{x}^j_T) F(\vec{x}^j_T;\vec{\theta})}
```

Here we assume that $`\epsilon(\vec{x}_T)`$  and $`R(\vec{x}_R,\vec{x}_T)`$ are uniform within the ith and jth bins of reconstructed and true variables, repectively. However, $`F(\vec{x}^j_T;\vec{\theta})`$ is the integration of $`f(\vec{x}_T;\vec{\theta})`$ in jth bin of true variable. 

# Code 

The code to run the fit takes the following arguments

```python
python LbToLclnu_fit.py -h 
```

```
usage: LbToLclnu_fit.py [-h] -f FLOATWC -s SEED [-b BSCHEME] [-n NEVNTS]
                        [-nf NFITS] [-sf SUFFIX] [-d DIREC] [-p PLOTRES]
                        [-effn EFFN] [-effp EFFPATH] [-resn RESN]
                        [-resp RESPATH] [-e FLOATED_FF [FLOATED_FF ...]]

Arguments for LbToLclnu_fit.py

optional arguments:
  -h, --help            show this help message and exit
  -f FLOATWC, --floatWC FLOATWC
                        (string) Name of the Wilson coefficient (WC) to be
                        floated. Available options are [CVR,CSR,CSL,CT].
  -s SEED, --seed SEED  (int) Seed for generation of fake/toy data. This
                        should be different for each toy.
  -b BSCHEME, --bscheme BSCHEME
                        (string) Binning scheme to be used. Available options
                        are [Scheme0,Scheme2,Scheme3,Scheme4,Scheme5,Scheme6]
                        and default is Scheme0.
  -n NEVNTS, --nevnts NEVNTS
                        (int) Size of the toy sample. Default is 7.5M events.
  -nf NFITS, --nfits NFITS
                        (int) Number of fits to conduct to a given sample.
                        Default in 1.
  -sf SUFFIX, --suffix SUFFIX
                        (int) A unique suffix added to the name of the fit
                        result file (*_suffix.txt) and plot file
                        (*_suffix.pdf). Default is 'toy'.
  -d DIREC, --direc DIREC
                        (string) Directory in which the fit result (.txt) and
                        plot is to be saved. Default in current directory.
  -p PLOTRES, --plotRes PLOTRES
                        (bool) Set to False if you do not want to plot the
                        result. Default is True.
  -effn EFFN, --effn EFFN
                        (bool) Set to True if you want efficiency included in
                        model. Default is False.
  -effp EFFPATH, --effpath EFFPATH
                        (string) Path to efficiency file. Default is: /disk/lh
                        cb_data/amathad/forHelena/ToyAngular/responsematrix_ef
                        f/Eff.p
  -resn RESN, --resn RESN
                        (bool) Set to True if you want resolution information
                        included in model. Default is False.
  -resp RESPATH, --respath RESPATH
                        (bool) Path to resoponse matrix file. Default is:/disk
                        /lhcb_data/amathad/forHelena/ToyAngular/responsematrix
                        _eff/responsematrix.p
  -e FLOATED_FF [FLOATED_FF ...], --floated_FF FLOATED_FF [FLOATED_FF ...]
                        (list) List of form factor (FF) parameters that you
                        want floated in the fit. Default is 'None' that is all
                        FF parameters are fixed. When CVR or CSR or CSL is set
                        as 'floatWC': 11 FF parameters can be floated, they
                        are a0f0 a0fplus a0fperp a1f0 a1fplus a1fperp a0g0
                        a0gplus a1g0 a1gplus a1gperp When CT is set as
                        'floatWC': In addition to the 11 FF, we can float 7
                        more which are a0hplus a0hperp a0htildeplus a1hplus
                        a1hperp a1htildeplus a1htildeperp
```

###  Examples for running fitting code  ###

Example 1

```python
python LbToLclnu_fit.py --floatWC CVR --seed 2
```

Options 'floatwc' and 'seed' are REQUIRED. All others are optional and if not provided will be set to the default values. 
The above example will float ``CVR`` in the fit and set the seed of generation of the toy to '2'. Each toy should have different seed. 

Example 2

```python
python LbToLclnu_fit.py -f CVR -s 2 -e a0f0 a1fplus
```

Note in this example that you can use either '--floatwc' or '-f' to specify the floated wilson coefficient.
The above example will float ``CVR`` in the fit, set the seed to 2 and float ``a0f0`` and ``a1fplus`` form parameters along with ``CVR``. The other options are set to the default ones. 

Example 3 

```python
python LbToLclnu_fit.py -f CVR -s 2 -e a0f0 a1fplus -d ./plots/ -sf toy -effn False -resn True -p False
```
The above example will 
- float ``CVR`` in the fit.
- set the seed to 2. 
- float ``a0f0`` and ``a1fplus`` form parameters along with ``CVR``, 
- write the results of the fit to the directory 'plots' from your current work directory. 
- add a suffix 'toy' to the file names that will be written to 'plots' directory.
- does not apply efficiency information to the model.
- applies resolution information to the model.
- does not plot the fit results but only writes the text file containing fit parameters. 

# Dependencies

The scripts depending on the following packages

AmpliTF: https://github.com/apoluekt/AmpliTF
TFA2: https://github.com/apoluekt/TFA2
