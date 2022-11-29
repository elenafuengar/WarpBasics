'''
cubcav.py

Example file to compute the longitudinal
wake potential and impedance for a cubic
cavity using the pre-computed fields from 
WarpX simulations

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
input_file = 'warpx.out'  #filename containing the beam information (optional)

# Field data
Ez_fname = 'Ez.h5'   #name of filename (optional)

# Output options
flag_save = True 
flag_plot = True

#---------------------------------------------------------

print('---------------------')
print('|   Running WAKIS   |')
print('---------------------')

user = wakis.Inputs.User(case = case, unit_m = unit_m, unit_t = unit_t, unit_f = unit_f)
beam = wakis.Inputs.Beam.from_WarpX(filename = input_file) 
field = wakis.Inputs.Field.from_WarpX(filename = input_file)

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

# Plot from save 
Wakis = wakis.Wakis.from_file()
fig, axs = Wakis.subplot()
