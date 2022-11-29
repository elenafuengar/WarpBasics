'''
reswall.py

Example file to compute the longitudinal
wake potential and impedance for a lossy
pipe using the pre-computed fields from 
CST simulations

@date: Created on 01.11.2022
@author: Elena de la Fuente
'''

import sys

# to impot wakis folder
sys.path.append('../../../')

import wakis

#---------------------------#
#       User variables      #
#---------------------------#

# abs path to input data (default is cwd)
# path = 'example/path/to/input/data/'

# set unit conversion 
unit_m = 1e-3  #default: mm
unit_t = 1e-9  #default: ns
unit_f = 1e9   #default: GHz

# Beam parameters 
sigmaz = 15*unit_m                  #beam longitudinal sigma [m]
q = 1e-9                          #beam charge in [C]
xsource, ysource = 0e-3, 0e-3     #beam center offset
xtest, ytest = 0e-3, 0e-3         #integration path offset

# Output options
flag_save = True 
flag_plot = True

#---------------------------------------------------------

print('---------------------')
print('|   Running WAKIS   |')
print('---------------------')

user = wakis.Inputs.User(unit_m = unit_m, unit_t = unit_t, unit_f = unit_f)
beam = wakis.Inputs.Beam.from_CST(q = q, sigmaz = sigmaz, 
                     xsource = xsource, ysource = ysource, 
                     xtest = xtest, ytest = ytest, chargedist = 'lambda.txt') 
field = wakis.Inputs.Field.from_CST(folder = '3d')

# Get data object
Wakis = wakis.Wakis.from_inputs(user, beam, field) 

# Run solver
Wakis.solve()

# Plot
if flag_plot:
    figs, axs = Wakis.plot()
    fig, axs = Wakis.subplot()

# Save
if flag_save:
    Wakis.save()

#----------------------------------------------------------
'''
# Plot from save 
Wakis = wakis.Wakis.from_file()
fig, axs = Wakis.subplot()
'''