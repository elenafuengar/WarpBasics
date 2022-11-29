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
sys.path.insert(1, '/mnt/c/Users/elefu/Documents/CERN/wakis/')

import wakis

#---------------------------#
#       User variables      #
#---------------------------#

# define EM solver: 'warpx' or 'cst'
case = 'cst'  

# abs path to input data (default is cwd)
# path = 'example/path/to/input/data/'

# set unit conversion 
unit_m = 1e-3  #default: mm
unit_t = 1e-9  #default: ns
unit_f = 1e9   #default: GHz

# Beam parameters 
input_file = 'warpx.inp'  #filename containing the beam information (optional)

# Field data
Ez_fname = 'Ez.h5'   #name of filename (optional)

# Output options
flag_save = True 
flag_plot = True

#---------------------------------------------------------

user = wakis.Inputs.User(case = case, unit_m = unit_m, unit_t = unit_t, unit_f = unit_f)
beam = wakis.Inputs.Beam.from_WarpX(filename = input_file) 
field = wakis.Inputs.Field.from_WarpX()

