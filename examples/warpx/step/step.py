'''
step.py
===
Example script to obtain wake potential and impedance
from 3d electromagnetic simulation output from warpx
using wakis module

How to use
---
1. `python warpx.py` to run EM simulation of the geometry defined 
   in `.stl` file
2. `ipython $example.py` to obtain the wake and impedance results

'''
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
input_file = 'warpx.json'         #filename containing the beam information (optional)
#q = 1e-9                          #beam charge in [C]
#sigmaz = 15                		  #beam longitudinal sigma [m]
#xsource, ysource = 0e-3, 0e-3     #beam center offset [m]
#xtest, ytest = 0e-3, 0e-3         #integration path offset [m]

# Field data
Ez_fname = 'Ez.h5'	 #name of filename (optional)
#Ez_folder = '3d/'   #relative path to folder containing multiple Ez files [CST](optional)

# Output options
flag_save = True 
flag_plot = True

#---------------------------#

# Initialize inputs
user = wakis.Inputs.User(case = case, unit_m = unit_m, unit_t = unit_t, unit_f = unit_f)
beam = wakis.Inputs.Beam.from_WarpX() #TODO
field = wakis.Inputs.Field.from_WarpX() #TODO

# Get data object
Wakis = wakis.from_inputs(user, beam, field) 

# Run solver
Wakis.solve()

# Plot
if flag_plot:
	figs, axs = Wakis.plot()
	fig, axs = Wakis.subplot()

# Save
if flag_save:
	Wakis.save(data, ext = 'json')
