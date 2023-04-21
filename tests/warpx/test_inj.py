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

# USER DEFINED VARIABLES

# path to geometry file
path=os.getcwd()+'/' #working directory
#path='$HOME/wakis/examples/XXX/'

#=======================#
# Simulation parameters #
#=======================#

CFL = 1.0               #Courant-Friedrichs-Levy criterion for stability
NUM_PROC = 1            #number of mpi processors wanted to use
UNIT = 1e-2             #conversion factor from input to [m]
Wake_length = 100*UNIT   #Wake potential length in s [m]

# beam parameters
q = 1e-9                   #Total beam charge [C]
sigmaz = 5*UNIT            #[m]

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
stl = True 

if stl:
# geometry defined by stl
    # Define stl file name
    stl_file = 'cubpillbox.stl'

    # STL geometry unit conversion to [m]
    stl_unit = 1.0e-2

    # Initialize WarpX EB object
    embedded_boundary = picmi.EmbeddedBoundary(stl_file=stl_file, stl_scale=stl_unit, stl_reverse_normal=True)

    # get simulation domain limits 
    xmin, xmax = -15.2*UNIT, +15.2*UNIT
    ymin, ymax = -15.2*UNIT, +15.2*UNIT
    zmin, zmax = -34.2*UNIT, +34.2*UNIT

else:
# geometry defined by implicit function
    
    plot_implicit_function = False 

    # Initialize WarpX EB object
    embedded_boundary = picmi.EmbeddedBoundary(
        implicit_function = "L=L_cavity; max(max(max(x-L/2,-L/2-x),max(y-L/2,-L/2-y)),max(z-L/2,-L/2-z))",
        #implicit_function = "L=L_cavity; max(max(x-L/2,-L/2-x),max(y-L/2,-L/2-y))",
        L_cavity = 64.*UNIT
    )

    xmin, xmax = -35*UNIT, +35*UNIT
    ymin, ymax = -35*UNIT, +35*UNIT
    zmin, zmax = -30*UNIT, +30*UNIT

    # *** geometry check ***
    if plot_implicit_function:
        sys.path.append('../../../source/')
        from helpers import *
        triang_implicit(implicit_function, BC=embedded_boundary, bbox=(zmin*1.2,zmax*1.2))
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
                                     #warpx_adjusted_pml = True, #crashes
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
    warpx_embedded_boundary = embedded_boundary,
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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def contourEz(x, y, Ez, title='Ez(x,y)', ax=None, fig=None):
    if ax is None:
        ax = fig.gca()
    ax.cla()
    Y, X = np.meshgrid(y,x)
    cm = ax.contourf(X, Y, Ez, cmap='jet')
    ax.set_title(title),
    ax.set_xlabel(f'{title[3]} [ncell]')
    ax.set_ylabel(f'{title[5]} [ncell]')   

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(cm, cax=cax, orientation='vertical')
    cbar.ax.locator_params(nbins=3)

def plotEz(z, Ez, rho=None, ax=None, fig=None, norm=True):
    if ax is None:
        ax = fig.gca()
    ax.cla()
    if norm:
        Ez = Ez/np.max(np.abs(Ez))
    ax.plot(z, Ez, c='g', lw=1.5, label='Ez')
    if rho is not None:
        rho = rho/np.max(rho)
        ax.plot(z, rho, c='r', lw=1.5, label='rho')

    ax.set_xlabel(f'z [ncell]')
    ax.set_ylabel(f'Ez [V/m]')
    ax.legend()
    
# Setup plotting flags

fcontour = False
fplot = True

plt.ion()
if fcontour:
    fig, axs = plt.subplots(1,3,tight_layout=True, figsize=(12,5)) 
if fplot:
    fig, ax = plt.subplots(1,1,tight_layout=True, figsize=(8,5)) 

# Start simulation --------------------------------------

print('[WARPX][PROGRESS] Starting simulation with a total of '+str(max_steps)+' timesteps...' )
t0 = time.time()
sim.step(100) 
for n in range(max_steps-100):
    sim.step(1)

    # Extract the electric field from all processors
    Ez = np.array(fields.EzWrapper().get_fabs(0,2,include_ghosts=False)[0])
    # Extract charge density
    rho = np.array(fields.JzWrapper().get_fabs(0,2,include_ghosts=False)[0]/(beam_beta*c))  #[C/m3]

    #arrays for plotting
    xf, yf, zf = np.arange(Ez.shape[0]), np.arange(Ez.shape[1]), np.arange(Ez.shape[2])

    if Ez.any() and fcontour:
        #Ez[np.abs(Ez) < 1e-4] = np.nan
        contourEz(xf, yf, Ez[:,:,int(len(zf)/2)], ax=axs[0], fig=fig, title='Ez(x,y)')
        contourEz(xf, zf, Ez[:,int(len(yf)/2), :], ax=axs[1], fig=fig, title='Ez(x,z)') 
        contourEz(yf, zf, Ez[int(len(xf)/2), :, :], ax=axs[2], fig=fig, title='Ez(y,z)')
        fig.suptitle(f'E field at timestep = {n}')
        fig.canvas.draw()
        fig.canvas.flush_events()

    if Ez.any() and fplot:
        plotEz(zf, Ez[int(len(xf)/2), int(len(yf)/2), :], rho[int(len(xf)/2), int(len(yf)/2), :], fig=fig)
        fig.suptitle(f'E field at timestep = {n}')
        fig.canvas.draw()
        fig.canvas.flush_events()

# Finish simulation --------------------------------------------

# Calculate simulation time
t1 = time.time()
totalt = t1-t0
print(f'[WARPX][PROGRESS] Simulation finished in {totalt} seconds' )


