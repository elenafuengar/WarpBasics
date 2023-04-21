# test for CST WP to Impedance FFT
import sys
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt
import wakis

#from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.ar_model import AutoReg
#from statsmodels.graphics.tsaplots import plot_pacf

wk = wakis.Wakis()

#bunch parameters
q = 1e-9
sigmaz = 18.737*1e-3

# read charge dist in distance lambda(z)
d1 = wk.read_cst_1d('lambda.txt')
chargedist = d1['Y'] 
z = d1['X']*1e-3

# read wake potential WPz(s)
d2 = wk.read_cst_1d('WP.txt')
WP = d2['Y']
s = d2['X']*1e-3

# interpolate charge dist to s
wk.calc_lambdas(z=z, s=s, chargedist=chargedist, q=q, sigmaz=sigmaz)
# calculate impedance with DFT 1000 samples
wk.calc_long_Z(WP=WP)

#TODO
# WPnew =