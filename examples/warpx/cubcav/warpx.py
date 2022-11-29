'''
Script to perform simulations of a step-like transition in WarpX 

How to use:
---
1. Define the path and simulation parameters
2. Define beam parameters
3. Define beam center 
4. Define integration path
5. Define geometry with STL file
6. Define mesh parameters

Run with:

 ```
 ipython 
 run warpx.py
 ```

Output
---
*Ez.h5*: Ez 3d matrix for every timestep in HDF5 file format
*warpx.inp*: Stores the geometry and simulation input in a dictionary with pickle 

Requirements
---
`numpy`, `scipy`, `stl`, `h5py`, `pywarpx`

'''

from pywarpx import picmi
from pywarpx import libwarpx, fields, callbacks
import numpy as np
import numpy.random as random
from scipy.constants import c, m_p, e
import pickle as pk
import time
import os
import sys
import h5py

from stl import mesh


t0 = time.time()

#-------------------------------------------------------------------------------
# USER DEFINED VARIABLES

# path to geometry file
path=os.getcwd()+'/' #working directory
#path='$HOME/wakis/examples/XXX/'

#=======================#
# Simulation parameters #
#=======================#

CFL = 1.0               #Courant-Friedrichs-Levy criterion for stability
NUM_PROC = 1            #number of mpi processors wanted to use
UNIT = 1e-3             #conversion factor from input to [m]
Wake_length = 1000*UNIT   #Wake potential length in s [m]

# flags
flag_logfile = False        #generates a .log file with the simulation info

# beam parameters
q = 1e-9                      #Total beam charge [C]
sigmaz = 15*UNIT            #[m]

# beam source center 
# ---[longitudinal impedance: beam center in 0,0]
# ---[dipolar impedance: beam center in a,0 or 0,a or a,a]
xsource = 0.0*UNIT
ysource = 0.0*UNIT 

# beam test center 
# ---[longitudinal impedance: test axis in 0,0]
# ---[quadrupolar impedance: test axis in a,0 or 0,a or a,a]
xtest = 0.0*UNIT   
ytest = 0.0*UNIT

#=====================#
# Geometry parameters #
#=====================#
stl = False 

if stl:
# geometry defined by stl
    # Define stl file name
    stl_file = 'longsymtaperCST.stl'

    # Initialize WarpX EB object
    embedded_boundary = picmi.EmbeddedBoundary(stl_file = path + stl_file)

    # STL geometry unit conversion to [m]
    stl_unit = 1e-3

    # read stl mesh
    obj = mesh.Mesh.from_file(path+stl_file)

    # get simulation domain limits 
    xmin, xmax = obj.x.min()*stl_unit, obj.x.max()*stl_unit
    ymin, ymax = obj.y.min()*stl_unit, obj.y.max()*stl_unit
    zmin, zmax = obj.z.min()*stl_unit, obj.z.max()*stl_unit 

else:
# geometry defined by implicit function
    
    plot_implicit_function = True 

    #define geometry
    w_cav, h_cav, L_cav = 50*UNIT, 50*UNIT, 30*UNIT
    w_pipe, h_pipe, L_pipe = 15*UNIT, 15*UNIT, 100*UNIT

    # Initialize WarpX EB object
    embedded_boundary = picmi.EmbeddedBoundary(
        implicit_function = "w=w_pipe+(w_cav-w_pipe)*(z<L_cav/2)*(z>-L_cav/2); h=h_pipe+(h_cav-h_pipe)*(z<L_cav/2)*(z>-L_cav/2); max(max(x-w/2,-w/2-x),max(y-h/2,-h/2-y))",
        w_cav = w_cav, 
        h_cav = h_cav, 
        L_cav = L_cav, 
        w_pipe = w_pipe, 
        h_pipe = h_pipe, 
    )

    xmin, xmax = -w_cav/2, +w_cav/2
    ymin, ymax = -h_cav/2, +h_cav/2
    zmin, zmax = -L_pipe/2, +L_pipe/2

    # *** geometry check ***
    if plot_implicit_function:
        sys.path.append('../../../source/')
        from helpers import *
        triang_implicit(implicit_function, BC=embedded_boundary, bbox=(zmin,zmax))
    # **********************


# Define mesh resolution in x, y, z
dh = 1.0*UNIT

# number of pml cells needed (included in domain limits)
n_pml = 10  #default: 10
flag_mask_pml = False  #removes the pml cells from the E field monitor

# define field monitor
xlo, xhi = xtest-2*dh, xtest+2*dh
ylo, yhi = ytest-2*dh, ytest+2*dh
zlo, zhi = zmin, zmax

#--------------------------------------------------------------------------------

#==================#
# Simulation setup #
#==================#

# get domain dimensions
W = (xmax - xmin)
H = (ymax - ymin)
L = (zmax - zmin)

# Define mesh cells per direction 
block_factor = 1 #mcm of nx, ny, nz
nx = int(W/dh)
ny = int(H/dh)
nz = int(L/dh)

# mesh cell widths
dx=(xmax-xmin)/nx
dy=(ymax-ymin)/ny
dz=(zmax-zmin)/nz

# mesh arrays (center of the cell)
x=np.linspace(xmin, xmax, nx+1)
y=np.linspace(ymin, ymax, ny+1)
z=np.linspace(zmin, zmax, nz)

# max grid size for mpi
max_grid_size_x = nx
max_grid_size_y = ny
max_grid_size_z = nz//NUM_PROC

print('[WARPX][INFO] Initialized mesh with ' + str((nx,ny,nz)) + ' number of cells')

# mask for the E field extraction
if flag_mask_pml:
    zmask = np.where((z >= zmin + n_pml*dz) & (z <= zmax - n_pml*dz))[0]
else:
    zmask = np.where((z >= zmin) & (z <= zmax))[0]
xmask = np.where((x >= xlo - dx) & (x <= xhi + dx))[0]
ymask = np.where((y >= ylo - dy) & (y <= yhi + dy))[0]

#Injection position [TO OPTIMIZE]
z_inj=zmin+5*dz
#z_inj=zmin+n_pml/2*dz

# generate the beam
bunch = picmi.Species(particle_type='proton',
                      name = 'beam')
# boundary conditions
lower_boundary_conditions = ['dirichlet', 'dirichlet', 'open']
upper_boundary_conditions = ['dirichlet', 'dirichlet', 'open']

# define grid
grid = picmi.Cartesian3DGrid(
    number_of_cells = [nx, ny, nz],
    lower_bound = [xmin, ymin, zmin],
    upper_bound = [xmax, ymax, zmax],
    lower_boundary_conditions = lower_boundary_conditions,
    upper_boundary_conditions = upper_boundary_conditions,
    lower_boundary_conditions_particles = ['absorbing', 'absorbing', 'absorbing'],
    upper_boundary_conditions_particles = ['absorbing', 'absorbing', 'absorbing'],
    moving_window_velocity = None,
    warpx_max_grid_size_x = max_grid_size_x,
    warpx_max_grid_size_y = max_grid_size_y,
    warpx_max_grid_size_z = max_grid_size_z,
    warpx_blocking_factor = block_factor,
)

flag_correct_div = False
flag_correct_div_pml = False
solver = picmi.ElectromagneticSolver(grid=grid, method='Yee', cfl=CFL,
                                     divE_cleaning = flag_correct_div,
                                     pml_divE_cleaning = flag_correct_div_pml,
                                     warpx_pml_ncell = n_pml,
                                     warpx_do_pml_in_domain = True,
                                     warpx_pml_has_particles = True,
                                     warpx_do_pml_j_damping = True, #Turned True for the pml damping
                                     #warpx_adjusted_pml = True, 
                                     )

# Obtain number of timesteps needed for the wake length
# time when the bunch enters the cavity
init_time = 8.53*sigmaz/c + (zmin+L/2)/c -z_inj/c #[s] injection time + PEC length - Injection length 

# timestep size
dt=CFL*(1/c)/np.sqrt((1/dx)**2+(1/dy)**2+(1/dz)**2)

# timesteps needed to simulate
max_steps=int((Wake_length+init_time*c+(zmax-zmin))/dt/c)

print('[WARPX][INFO] Timesteps to simulate = '+ str(max_steps) + ' with timestep dt = ' + str(dt))
print('[WARPX][INFO] Wake length = '+str(Wake_length/UNIT)+ ' mm')

sim = picmi.Simulation(
    solver = solver,
    max_steps = max_steps,
    warpx_embedded_boundary=embedded_boundary,
    particle_shape = 'cubic', 
    verbose = 1
)

beam_layout = picmi.PseudoRandomLayout(n_macroparticles = 0)

sim.add_species(bunch, layout=beam_layout)

sim.initialize_inputs()

#==========================#
# Setup the beam injection #
#==========================#

# beam sigma in time and longitudinal direction
# defined by the user

# transverse sigmas.
sigmax = 2e-4
sigmay = 2e-4

# spacing between bunches
b_spac = 25e-9
# offset of the bunch centroid
t_offs = -init_time #+ 9*dz/c   #like CST (-160 mm) + non-vacuum cells from injection point
# number of bunches to simulate
n_bunches = 1

# beam energy
beam_gamma = 479.
beam_uz = beam_gamma*c
beam_beta = np.sqrt(1-1./(beam_gamma**2))

# macroparticle info
N=10**7
bunch_charge = q #beam charge in [C] defined by the user
bunch_physical_particles  = int(bunch_charge/e)
bunch_macro_particles = N
bunch_w = bunch_physical_particles/bunch_macro_particles

# Define the beam offset
ixsource=int((xsource-x[0])/dx)
iysource=int((ysource-y[0])/dy)
print('[WARPX][INFO] Beam center set to ('+str(round(x[ixsource]/UNIT,3))+','+str(round(y[iysource]/UNIT,3))+',z,t) [mm]')

bunch_rms_size            = [sigmax, sigmay, sigmaz]
bunch_rms_velocity        = [0.,0.,0.]
bunch_centroid_position   = [xsource, ysource, z_inj] #Always inject in position 5
bunch_centroid_velocity   = [0.,0.,beam_uz]

# time profile of a gaussian beam
def time_prof(t):
    val = 0
    sigmat =  sigmaz/c
    dt = libwarpx.libwarpx_so.warpx_getdt(0)
    for i in range(0,n_bunches):
        val += bunch_macro_particles*1./np.sqrt(2*np.pi*sigmat*sigmat)*np.exp(-(t-i*b_spac+t_offs)*(t-i*b_spac+t_offs)/(2*sigmat*sigmat))*dt
    return val

# auxiliary function for injection
def nonlinearsource():
    t = libwarpx.libwarpx_so.warpx_gett_new(0)
    NP = int(time_prof(t))
    if NP>0:
        x = random.normal(bunch_centroid_position[0],bunch_rms_size[0],NP)
        y = random.normal(bunch_centroid_position[1],bunch_rms_size[1],NP)
        z = bunch_centroid_position[2]

        vx = np.zeros(NP)
        vy = np.zeros(NP)
        vz = np.ones(NP)*c*np.sqrt(1-1./(beam_gamma**2))

        beam_beta = np.sqrt(1-1./(beam_gamma**2))
        
        ux = np.zeros(NP)
        uy = np.zeros(NP)
        uz = beam_beta * beam_gamma * c

        libwarpx.add_particles(
            species_name='beam', x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, w=bunch_w*np.ones(NP),
        )

callbacks.installparticleinjection(nonlinearsource)

print('[WARPX][INFO] Finished simulation setup')

#================#
# Run simulation #
#================#
t0 = time.time()

if flag_logfile:
    # Create logfile
    sys.stdout = open(path+"log.txt", "w")

# Create Ez.h5 files overwriting previous one
hf_name='Ez.h5'
if os.path.exists(path+hf_name):
    os.remove(path+hf_name)

hf_Ez = h5py.File(path+hf_name, 'w')

# Define the integration path for test particle (xtest, ytest)
ixtest=int((xtest-x[0])/dx)
iytest=int((ytest-y[0])/dy)
print('[WARPX][INFO] Field will be extracted around ('+str(round(x[ixtest]/UNIT,3))+','+str(round(y[iytest]/UNIT,3))+',z,t) [mm]')

# Define number for datasets title
prefix=[]
for n_step in range(1, max_steps):
    prefix.append('0'*(5-int(np.log10(n_step))))

prefix=np.append('0'*5, prefix)

# Step by step running + saving data in hdf5 format
t=[]
rho_t=[]

# Start simulation --------------------------------------

print('[WARPX][PROGRESS] Starting simulation with a total of '+str(max_steps)+' timesteps...' )
for n_step in range(max_steps):

    print(n_step)
    sim.step(1)

    # Extract the electric field from all processors
    Ez = fields.EzWrapper().get_fabs(0,2,include_ghosts=False)[0]
    # Extract charge density
    rho = fields.JzWrapper().get_fabs(0,2,include_ghosts=False)[0]/(beam_beta*c)  #[C/m3]
    # Extraxt the timestep size
    dt = libwarpx.libwarpx_so.warpx_getdt(0)

    # append to t, rho lists
    t.append(n_step*dt)
    rho_t.append(rho[ixsource,iysource,:])

    # Save the 3D Ez matrix into a hdf5 dataset
    #hf_Ez.create_dataset('Ez_'+prefix[n_step]+str(n_step), data=Ez[:,:,:])

    # Saves the Ez field in a prism along the z axis 3 cells wide into a hdf5 dataset
    hf_Ez.create_dataset('Ez_'+prefix[n_step]+str(n_step), data=Ez[xmask][:,ymask][:,:,zmask])

# Finish simulation --------------------------------------------

# Calculate simulation time
t1 = time.time()
totalt = t1-t0
print('[WARPX][PROGRESS] Run terminated in %ds' %totalt)

# Close the hdf5 files
hf_Ez.close()

#Create np.arrays
rho_t=np.transpose(np.array(rho_t)) #(z,t)
t=np.array(t)

# Get linear charge distribution
rho=rho_t*dx*dy                             #rho(z,t) [C/m]
tmax=np.argmax(rho[len(rho[:,0])//2, :])    #t where rho is max at cavity center
qz=np.sum(rho[:,tmax])*dz                   #charge along the z axis
chargedist = rho[:,tmax]*q/qz               #total charge in the z axis lambda(z) [C/m]

if flag_logfile:
    # Close logfile
    sys.stdout.close()

#=================#
# Generate output # 
#=================#

# Create dictionary with input data. SI UNITs: [m], [s], [C]
data = { 't' : t,
         'x' : x[xmask],   
         'y' : y[ymask],
         'z' : z[zmask],
         'sigmaz' : sigmaz,
         'xsource' : xsource,
         'ysource' : ysource,
         'xtest' : xtest,
         'ytest' : ytest,
         'q' : q, 
         'chargedist' : {'X' : z[zmask], 'Y' : chargedist},
         'rho' : rho,
         'unit_m' : UNIT,
         'x0' : x,
         'y0' : y,
         'z0' : z
        }

# write the input dictionary to a txt using pickle module
with open(path+'warpx.out', 'wb') as fp:
    pk.dump(data, fp)

print('[WARPX][! OUT] out file succesfully generated') 
