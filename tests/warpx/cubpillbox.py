'''
Cubic cavity simulation using
Wakis API for WarpX simulations
'''
import sys
sys.path.append('../../')

import numpy as np
#import matplotlib.pyplot as plt
import wakis

em = wakis.WarpX()
unit = 1.0e-3

# Beam
em.q = 1e-9
em.sigmaz = 18*unit 				#for f_max ~2 GHz
em.xsource, em.ysource = 0.0, 0.0
em.xtest, em.ytest = 0.0, 0.0

# Geometry
em.stl_file = 'cubpillbox.stl'
em.stl_scale = 1e-3

# Mesh
em.nx = 32
em.ny = 32
em.nz = 83

# Simulation
em.wakelength = 1000*unit
em.verbose = 1
em.simulation_setup()

# Test EB and Injection
#em.testEB()
em.testInj(nplots=4)

# Run simulation and save field in hdf5
#em.field_monitor(nx=1, ny=1, mask_pml=True)
#em.run(hfd5=True)