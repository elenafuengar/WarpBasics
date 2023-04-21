'''
Test for longitudinal wake 
potential and impedance
'''

import sys
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt
import wakis

# generate h5files
#wk = wakis.Wakis()
#wk.read_cst_3d(path='/mnt/d/data/3d_default/', filename='Ez.h5')
#wk.read_cst_3d(path='/mnt/d/data/3d_dipolar/', filename='Ez_dipolar.h5')
#wk.read_cst_3d(path='/mnt/d/data/3d_quadrupolar/', filename='Ez_quadrupolar.h5')
#wk.read_cst_3d(path='/eos/user/e/edelafue/data/cubpillbox/3d/', filename='Ez.h5', units=1e-2)

# read files
path = '/eos/user/e/edelafue/data/cubpillbox/3d/'
wk = wakis.Wakis(Ez_file=path+'Ez.h5', q=1e-9, sigmaz=5e-2, xsource=0.0, ysource=0.0)

#path = '/eos/user/e/edelafue/data/cubcav/'
#wk = wakis.Wakis(Ez_file=path+'Ez.h5', q=1e-9, sigmaz=18e-3, xsource=0.0, ysource=0.0)

wk.calc_long_WP()
wk.plot_long_WP(compare=True, units=1e-2)

#read CST
cst = wk.read_cst_1d('WP.txt')
WP = cst['Y']
s = cst['X']*1e-2