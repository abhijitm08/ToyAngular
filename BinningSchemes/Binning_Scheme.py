import numpy as np

def get_bin_edges(xedges, yedges):
    xbw           = xedges[1:] - xedges[:-1]
    xbin_centers  = xedges[:-1] + 0.5 * xbw
    ybw           = yedges[1:] - yedges[:-1]
    ybin_centers  = yedges[:-1] + 0.5 * ybw
    bin_limits = {}
    bin_centers= {}
    Binnum     = 0
    for i in range(xedges.shape[0]):
        for j in range(yedges.shape[0]):
            if i == (xedges.shape[0]-1) or j == (yedges.shape[0]-1): break
            bin_limits['Bin'+str(Binnum)]  = [(xedges[i], xedges[i+1]), (yedges[j], yedges[j+1])]
            bin_centers['Bin'+str(Binnum)] = [xbin_centers[i], ybin_centers[j]]
            Binnum += 1

    #print(bin_limits)
    #print(bin_centers)
    return bin_centers, bin_limits
    
def defing_binning_scheme():
    mLb     = 5619.49997776e-3    #GeV
    mLc     = 2286.45992749e-3    #GeV
    mlep    = 105.6583712e-3      #GeV Mu

    #Note: It is NECESSARY to include min and max of qsq and cthl when putting the bin edges!!!

    #Scheme 0: 4x3 = 12
    BinScheme = {}
    bscheme = 'Scheme0'
    BinScheme[bscheme] = {}
    BinScheme[bscheme]['qsq']  = np.linspace(mlep**2, (mLb - mLc)**2, 4+1)
    BinScheme[bscheme]['cthl'] = np.linspace(-1., 1., 3+1)

    #Scheme 1: 6x4 = 24
    bscheme = 'Scheme1'
    BinScheme[bscheme] = {}
    BinScheme[bscheme]['qsq']  = np.array([ mlep**2, 0.5, 1.8, 5., 8.5, 9.7, (mLb - mLc)**2])
    BinScheme[bscheme]['cthl'] = np.array([-1. , -0.4, 0. , 0.4, 1. ])

    #Scheme 2: 5x5 = 25
    bscheme = 'Scheme2'
    BinScheme[bscheme] = {}
    BinScheme[bscheme]['qsq']  = np.array([mlep**2, 2.23076215, 4.45036061, 6.66995906, 8.88955752, (mLb - mLc)**2])
    BinScheme[bscheme]['cthl'] = np.array([-1. , -0.6, -0.2, 0.2, 0.6, 1. ])

    #Scheme 3: 7x5 = 35
    bscheme = 'Scheme3'
    BinScheme[bscheme] = {}
    BinScheme[bscheme]['qsq']  = np.array([ mlep**2, 0.5, 2., 4.5, 7.5, 9., 10., (mLb - mLc)**2])
    BinScheme[bscheme]['cthl'] = np.array([-1. , -0.75, -0.5 , 0. , 0.5 , 1. ])

    #Scheme 4: 7x5 = 35
    bscheme = 'Scheme4'
    BinScheme[bscheme] = {}
    BinScheme[bscheme]['qsq']  = np.array([ mlep**2, 1.59659116, 3.18201863, 4.7674461 , 6.35287357, 7.93830104, 9.52372851, (mLb - mLc)**2])
    BinScheme[bscheme]['cthl'] = np.array([-1. , -0.6, -0.2, 0.2, 0.6, 1. ])

    #Scheme 5: 7x6 = 42
    bscheme = 'Scheme5'
    BinScheme[bscheme] = {}
    BinScheme[bscheme]['qsq']  = np.array([ mlep**2, 0.8, 2.4, 4.8, 7.8, 8.9, 9.8, (mLb - mLc)**2])
    BinScheme[bscheme]['cthl'] = np.array([-1. , -0.9, -0.7, -0.5, -0.1, 0.4, 1. ])

    #Scheme 6: 40x40 = 1600
    bscheme = 'Scheme6'
    BinScheme[bscheme] = {}
    BinScheme[bscheme]['qsq']  = np.linspace(mlep**2, (mLb - mLc)**2, 40+1)
    BinScheme[bscheme]['cthl'] = np.linspace(-1., 1., 40+1)

    #Scheme 7: 30x30 = 900
    bscheme = 'Scheme7'
    BinScheme[bscheme] = {}
    BinScheme[bscheme]['qsq']  = np.linspace(mlep**2, (mLb - mLc)**2, 30+1)
    BinScheme[bscheme]['cthl'] = np.linspace(-1., 1., 30+1)


    #calculate bin centers and bin limit dictionaries
    for k in list(BinScheme.keys()):
        bcnts, bedges = get_bin_edges(BinScheme[k]['qsq'], BinScheme[k]['cthl'])
        BinScheme[k]['bin_centers'] = bcnts
        BinScheme[k]['bin_limits' ] = bedges

    return BinScheme
