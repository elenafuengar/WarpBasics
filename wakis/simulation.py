'''
Class to run WarpX simulations
with optimized parameters for 
accelerator structures

Requirements
------------
pywarpx
numpy-stl (optional)

@date: Created on 14.03.2023
@author: Elena de la Fuente
'''


import pickle as pk
import time
import os
import sys

#dependencies
import h5py
import numpy as np
import matplotlib.pyplot as plt

from wakis.logger import get_logger
from tqdm import tqdm

# WarpX
try:    
    from pywarpx import picmi
    from pywarpx import libwarpx, fields, callbacks
except:
    print('pywarpx not imported')

# time profile of a gaussian beam
def time_prof(t):
    val = 0
    global _sigmaz, _N, _b_spac, _t_inj

    sigmat = sigmaz/picmi.constants.c
    dt = libwarpx.libwarpx_so.warpx_getdt(0)
    for i in range(0,n_bunches):
        val += _N*1./np.sqrt(2*np.pi*sigmat*sigmat)*np.exp(-(t-i*_b_spac-_t_inj)*(t-i*_b_spac-_t_inj)/(2*sigmat*sigmat))*dt
    return val

# auxiliary function for injection
def nonlinearsource():
    global _beam_beta, _beam_gamma, _bunch_w

    t = libwarpx.libwarpx_so.warpx_gett_new(0)
    NP = int(time_prof(t))
    if NP>0:
        x = np.random.normal(bunch_centroid_position[0],bunch_rms_size[0],NP)
        y = np.random.normal(bunch_centroid_position[1],bunch_rms_size[1],NP)
        z = bunch_centroid_position[2]

        vx = np.zeros(NP)
        vy = np.zeros(NP)
        vz = np.ones(NP)*picmi.constants.c*_beam_beta
        
        ux = np.zeros(NP)
        uy = np.zeros(NP)
        uz = _beam_beta*_beam_gamma*picmi.constants.c

        libwarpx.add_particles(
            species_name='beam', x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, w=_bunch_w*np.ones(NP),
        )

class WarpX():
    '''Class to run optimized WarpX simulations
    '''
    def __init__(self, wakelength=1., sigmaz=5e-2, xsource=0., ysource=0.,
                 ytest=0., xtest=0., q=1e9, stl_file=None, stl_scale=1.0, dh=None, 
                 setup=False, log_level=1, **kwargs):

        '''WarpX wrapper to run wakefield simulations

        Inputs expected in S.I. units: dimensions [m], time [s], charge [C]
        '''

        self.log = get_logger(level=log_level) #verbose level: 1: DEBUG, 2: INFO, 3:WARNING, 0:DEFAULT (INFO)

        # Constants (form SciPy.constants)
        self.c = picmi.constants.c #299792458.0
        self.e = picmi.constants.q_e #1.602176634e-19
        self.m_p = picmi.constants.m_p #1.67262192369e-27

        # geometry
        self.stl_file = stl_file
        self.stl_scale = stl_scale
        self.stl_reverse_normal = True #'True' for vacuum geometry, 'False' for conductors geometry
        self.implicit_function = None
        
        # domain
        self.xmin, self.xmax = None, None
        self.ymin, self.ymax = None, None 
        self.zmin, self.zmax = None, None
        self.factor_x = 1.11 #extend domain to enclose stl
        self.factor_y = 1.11 #extend domain to enclose stl
        self.factor_z = 0.95 #makes sure vacuum enters pml

        # mesh 
        self.nx = None
        self.ny = None
        self.nz = None

        self.dx = None
        self.dy = None
        self.dz = None
        self.dh = dh

        # simulation
        self.CFL = 1.0
        self.verbose = 0 #set 0 to avoid verbose outputs
        self.wakelength = wakelength
        self.n_pml = 10
        self.block_factor = 1
        self.zinj = None

        # beam
        self.q = q
        self.sigmaz = sigmaz
        self.xsource, self.ysource = xsource, ysource
        self.xtest, self.ytest = xtest, ytest
        self.particle_shape = 'cubic'

        #beam injection
        self.sigmax = 2e-4
        self.sigmay = 2e-4
        self.b_spac = 25e-9 
        self.n_bunches = 1
        self.t_inj = None

        # beam energy
        self.beam_gamma = 479.
        self.beam_uz = self.beam_gamma*self.c
        self.beam_beta = np.sqrt(1-1./(self.beam_gamma**2))

        # macroparticle info
        self.bunch_charge = self.q #beam charge in [C] defined by the user
        self.bunch_macro_particles = 10**7
        
        #field monitor
        self.xmask = None
        self.ymask = None
        self.zmask = None
        self.mask_pml = False

        # outs
        self.Ez = None       # Ez(x,y,z)   [V/m]
        self.rho = None      # rho(x,y,z)  [C/m3]
        self.Ezt = None      # Ez(x_s, y_s, z, t)  [V/m]
        self.lambdat = None  # rho(x_s, y_s, z, t)*dx*dy  [C/m]
        self.t = None 

        #handle kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

        if setup:
            self.simulation_setup()

    def set_geometry(self, stl_file=None, stl_scale=None, 
                    stl_reverse_normal=None, domain_from_stl=False, 
                    implicit_function=None, **kwargs):

        if stl_file is not None:
            self.stl_file = stl_file
        if stl_scale is not None:
            self.stl_scale = stl_scale
        if stl_reverse_normal is not None:
            self.stl_reverse_normal = stl_reverse_normal

        if self.stl_file is not None:
            self.embedded_boundary = picmi.EmbeddedBoundary(stl_file=self.stl_file,     
                        stl_scale=self.stl_scale, stl_reverse_normal=self.stl_reverse_normal)
        
        elif self.implicit_function is not None:
            self.embedded_boundary = picmi.EmbeddedBoundary(implicit_function=self.implicit_function, **kwargs)

        #handle kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

        if domain_from_stl:
            self.set_domain_from_stl()
        else:
            self.set_domain()

    def set_domain(self, xmin=None, xmax=None, ymin=None, 
                   ymax=None, zmin=None, zmax=None, **kwargs):

        #handle kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

        if not [x for x in (xmin, xmax, ymin, ymax, zmin, zmax) if x is None]:

            self.xmin, self.xmax = xmin, xmax
            self.ymin, self.ymax = ymin, ymax 
            self.zmin, self.zmax = zmin, zmax

            self.W = self.xmax-self.xmin
            self.H = self.ymax-self.ymin
            self.L = self.zmax-self.zmin

        else: 
            self.set_domain_from_stl(**kwargs)


    def set_domain_from_stl(self, **kwargs):

        #handle kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

        #needs numpy-stl
        try:
            from stl import mesh
        except:
            self.log.warning('numpy-stl could not be imported')
            self.log.warning('Set valid domain limits or install numpy-stl')
            raise ImportError('numpy-stl could not be imported')

        if self.stl_file is None:
            self.log.warning('Stl file is not specified')
            raise Exception('Stl file is not specified')

        #get domain limits from stl
        else:
            obj = mesh.Mesh.from_file(self.stl_file)

            # get simulation domain limits 
            self.xmin = obj.x.min()*self.factor_x*self.stl_scale 
            self.xmax = obj.x.max()*self.factor_x*self.stl_scale
            self.ymin = obj.y.min()*self.factor_y*self.stl_scale 
            self.ymax = obj.y.max()*self.factor_y*self.stl_scale
            self.zmin = obj.z.min()*self.factor_z*self.stl_scale
            self.zmax = obj.z.max()*self.factor_z*self.stl_scale

            self.W = self.xmax-self.xmin
            self.H = self.ymax-self.ymin
            self.L = self.zmax-self.zmin


    def set_mesh(self, **kwargs):

        if not [x for x in (self.nx, self.ny, self.nz) if x is None]:
            self.dx=(self.W)/self.nx
            self.dy=(self.H)/self.ny
            self.dz=(self.L)/self.nz

        elif not [x for x in (self.dx, self.dy, self.dz) if x is None]:
            self.nx = int(self.W/self.dx)
            self.ny = int(self.H/self.dy)
            self.nz = int(self.L/self.dz)

        elif self.dh is not None:
            self.nx = int(self.W/self.dh)
            self.ny = int(self.H/self.dh)
            self.nz = int(self.L/self.dh)

            self.dx, self.dy, self.dz = dh, dh, dh

        else: 
            self.log.warning('Specify either number of cells or mesh step for x, y, z')

        # mesh arrays (center of the cell)
        self.x = np.linspace(self.xmin, self.xmax, self.nx+1)
        self.y = np.linspace(self.ymin, self.ymax, self.ny+1)
        self.z = np.linspace(self.zmin, self.zmax, self.nz)

        #handle kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

        lower_boundary_conditions = ['dirichlet', 'dirichlet', 'open']
        upper_boundary_conditions = ['dirichlet', 'dirichlet', 'open']

        max_grid_size_x = self.nx
        max_grid_size_y = self.ny
        max_grid_size_z = self.nz

        # define grid
        self.grid = picmi.Cartesian3DGrid(
            number_of_cells = [self.nx, self.ny, self.nz],
            lower_bound = [self.xmin, self.ymin, self.zmin],
            upper_bound = [self.xmax, self.ymax, self.zmax],
            lower_boundary_conditions = lower_boundary_conditions,
            upper_boundary_conditions = upper_boundary_conditions,
            lower_boundary_conditions_particles = ['absorbing', 'absorbing', 'absorbing'],
            upper_boundary_conditions_particles = ['absorbing', 'absorbing', 'absorbing'],
            moving_window_velocity = None,
            warpx_max_grid_size_x = max_grid_size_x,
            warpx_max_grid_size_y = max_grid_size_y,
            warpx_max_grid_size_z = max_grid_size_z,
            warpx_blocking_factor = self.block_factor,
        )    

        if self.verbose:
            self.log.info('Initialized mesh with ' + str((self.nx, self.ny, self.nz)) + ' number of cells')

    def set_solver(self, wakelength=None, max_steps=None, n_pml=None, sigmaz=None,
                   CFL=None, particle_shape=None, verbose=None, **kwargs):

        self.flag_correct_div = False
        self.flag_correct_div_pml = False

        if n_pml is not None: self.n_pml = n_pml
        if sigmaz is not None: self.sigmaz = sigmaz
        if wakelength is not None: self.wakelength = wakelength

        #set injection position
        self.z_inj = self.zmin+5*self.dz #self.n_pml/2*self.dz

        # time when the beam enters the domain
        nsigmas = 8.548921333333334 #factor used in CST studio
        self.t_inj = nsigmas*self.sigmaz/self.c - (self.z_inj-self.zmin)/self.c #[s] injection time - Injection length 

        # timestep size
        if CFL is not None: self.CFL=CFL
        self.dt=self.CFL*(1/self.c)/np.sqrt((1/self.dx)**2+(1/self.dy)**2+(1/self.dz)**2)

        # timesteps needed to simulate based on the wakelength
        if max_steps is not None:
            self.max_steps = max_steps
        else:
            self.max_steps=int((self.wakelength+self.t_inj*self.c+(self.zmax-self.zmin))/self.dt/self.c)

        #handle kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

        self.solver = picmi.ElectromagneticSolver(grid=self.grid, method='Yee', cfl=self.CFL,
                                             divE_cleaning = self.flag_correct_div,
                                             pml_divE_cleaning = self.flag_correct_div_pml,
                                             warpx_pml_ncell = self.n_pml,
                                             warpx_do_pml_in_domain = True,
                                             warpx_pml_has_particles = True,
                                             warpx_do_pml_j_damping = True, #Turned True for the pml damping
                                             )

        self.sim = picmi.Simulation(
            solver = self.solver,
            max_steps = self.max_steps,
            warpx_embedded_boundary = self.embedded_boundary,
            particle_shape = self.particle_shape, 
            verbose = self.verbose
        )

        # add proton beam to simulation
        bunch = picmi.Species(particle_type='proton', name='beam')
        beam_layout = picmi.PseudoRandomLayout(n_macroparticles = 0)
        self.sim.add_species(bunch, layout=beam_layout)
        self.sim.initialize_inputs()

        if self.verbose:
            self.log.info(f'Timesteps to simulate = {self.max_steps} with timestep dt = {self.dt} s')
            self.log.info(f'Wake length = {self.wakelength} m')


    def set_beam(self, **kwargs):

        # offset of the bunch centroid
        self.bunch_physical_particles = int(self.bunch_charge/self.e)
        self.bunch_w = self.bunch_physical_particles/self.bunch_macro_particles

        # Define the beam offset
        self.ixsource=int((self.xsource-self.x[0])/self.dx)
        self.iysource=int((self.ysource-self.y[0])/self.dy)

        bunch_rms_size            = [self.sigmax, self.sigmay, self.sigmaz]
        bunch_rms_velocity        = [0.,0.,0.]
        bunch_centroid_position   = [self.xsource, self.ysource, self.z_inj] #Always inject in position 5
        bunch_centroid_velocity   = [0.,0.,self.beam_uz]

        #handle kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

        _sigmaz, _N, _b_spac, _t_inj = self.sigmaz, self.bunch_macro_particles, self.b_spac, self.t_inj
        global _sigmaz, _N, _b_spac, _t_inj

        _beam_beta, _beam_gamma, _bunch_w = self.beam_beta, self.beam_gamma, self.bunch_w
        global _beam_beta, _beam_gamma, _bunch_w

        # install injection
        callbacks.installparticleinjection(nonlinearsource)

        if self.verbose:
            self.log.info(f'Added particle beam with center at: x = {self.x[self.ixsource]}, y = {self.y[self.iysource]}')

    def simulation_setup(self, **kwargs):
    
        # set geometry (EB) -> domain -> domain_from_stl
        self.set_geometry(**kwargs)

        # set mesh (grid)
        self.set_mesh(**kwargs)

        # set solver (solver, sim)
        self.set_solver(**kwargs)

        # set beam (callback)
        self.set_beam(**kwargs)

        if self.verbose:
            self.log.info('Finished simulation setup')

    def field_monitor(self, xtest=None, ytest=None, nx=1, ny=1, nz=None, xlo=None, 
                      xhi=None, ylo=None, yhi=None, mask_pml=True):
        '''Field monitor around xtest, ytest

        '''

        if xtest is not None: self.xtest = xtest
        if ytest is not None: self.ytest = ytest

        # x
        if xhi is not None and xlo is not None:
            self.xmask = np.where((self.x >= self.xtest - xlo) & (self.x <= self.xtest + xhi))[0]
        else:
            self.xmask = np.where((self.x > self.xtest - (nx+1)*self.dx) & (self.x < self.xtest + (nx+1)*self.dx))[0]

        # y
        if yhi is not None and ylo is not None:
            self.ymask = np.where((self.y >= self.ytest - ylo) & (self.y <= self.ytest + yhi))[0]
        else:
            self.ymask = np.where((self.y > self.ytest - (ny+1)*self.dy) & (self.y < self.ytest + (ny+1)*self.dy))[0]

        # z
        if mask_pml:
            self.zmask = np.where((self.z >= self.zmin + self.n_pml*self.dz) & (self.z <= self.zmax - self.n_pml*self.dz))[0]
        elif nz is not None:
            self.zmask = np.where((self.z <= self.z[self.nz//2] + nz*self.dz) & (self.z >= self.z[self.nz//2] - nz*self.dz))[0]
        else:
            self.zmask = np.where((self.z >= self.zmin) & (self.z <= self.zmax))[0]


    def run(self, hdf5=True, hf_name='Ez.h5', **kwargs):

        t, lambdat, Ezt = [], [], []

        # Create hdf5
        if hdf5:
            # Create Ez.h5 files overwriting previous one
            if os.path.exists(hf.name):
                os.remove(hf_name)
            hf = h5py.File(hf_name, 'w')
            # Define number for datasets title
            prefix=[]
            for n in range(1, self.max_steps):
                prefix.append('0'*(5-int(np.log10(n))))

            prefix=np.append('0'*5, prefix)

        # Control output
        if self.verbose: 
            loop = range(self.max_steps)
        else:
            loop = tqdm(range(self.max_steps), 'Running WarpX simulation...')

        for n in loop:

            self.sim.step(1)

            # Extract the electric field from all processors
            Ez = fields.EzWrapper().get_fabs(0,2,include_ghosts=False)[0]
            # Extract charge density
            rho = fields.JzWrapper().get_fabs(0,2,include_ghosts=False)[0]/(self.beam_beta*self.c)  #[C/m3]
            # Extract the timestep size
            dt = libwarpx.libwarpx_so.warpx_getdt(0)

            #mask
            Ez = Ez[self.xmask][:,self.ymask][:,:,self.zmask]
            rho = rho[self.xmask][:,self.ymask][:,:,self.zmask]

            #append
            t.append(n*dt) #time array
            lambdat.append(rho[self.ixsource, self.iysource, :]*self.dx*self.dy) #line charge density (x_s, y_s, z, t) [C/m]
            Ezt.append(Ez[self.ixsource, self.iysource, :]) #Ez field (x_s, y_s, z, t) [V/m]

            #save
            if hdf5:
                hf.create_dataset('Ez_'+prefix[n]+str(n), data=Ez)


        # Get linear charge distribution
        t_max=np.argmax(lambdat[len(self.zmask)//2, :])      #max at cavity center
        qz=np.sum(lambdat[:,t_max])*self.dz     #total charge along the z axis

        self.lambdat = np.array(lambdat)*self.q/qz        #total charge lumped into z axis
        self.Ezt = np.array(Ezt)
        self.t = np.array(t)

        hf['x'] = self.x[self.xmask]
        hf['y'] = self.y[self.ymask]
        hf['z'] = self.z[self.zmask]
        hf['t'] = self.t

        hf.close()


    def step(self, n=1):

        self.sim.step(n)
        self.Ez = fields.EzWrapper().get_fabs(0,2,include_ghosts=False)[0] #[V/m]
        self.rho = fields.JzWrapper().get_fabs(0,2,include_ghosts=False)[0]/(self.beam_beta*self.c)  #[C/m3]


    def save(self, filename='warpx', ext='dat'):

        '''
        Save results in 'warpx' file. 
        Two dumping methods supported: 'json' and 'pickle'

        Parameters
        ----------
        filename: str, default 'warpx'
            Name of output file
        ext :  str, default 'dat'
            Extention to be used in output file 'json' or 'pickle'

        Raises
        ------
        Warning
            If the file extension is not supported
        '''

        self.xf = self.x[self.xmask]
        self.yf = self.y[self.ymask]
        self.zf = self.z[self.zmask]
        self.chargedist = self.lambdat[:, np.argmax(lambdat[len(self.zmask)//2, :])]

        d = self.__dict__
        keys = ['Ezt', 'lambdat', 'chargedist', 't', 'x', 'y', 'z', \
                'lambdat', 'xsource', 'ysource', 'xtest', 'ytest',  \
                'q', 'sigmaz', 't_inj', 'wakelength', 'xf', 'yf', 'zf']

        exts = ['pk', 'pickle', 'out', 'dat', 'wpx']

        if ext == 'json':
            j = json.dumps({k: d[k].tolist() for k in keys})
            with open(filename + '.' + ext, 'w') as f:
                json.dump(j, f)
            self.log.info('"' + filename + '.' + ext +'" file succesfully generated') 

        elif ext in exts:
            p = {k: d[k] for k in keys}
            with open(filename + '.' + ext, 'wb') as f:
                pk.dump(p, f)
            self.log.info('"' + filename + '.' + ext +'" file succesfully generated') 
        
        else: 
            self.log.warning(f'Extension ".{ext}" not supported, choose one of the following: {exts}')

    def testEB(self, n=1, dim=2, save=False, ext='png'):
        print('hi')
        self.sim.step(n)

        label = ['x', 'y', 'z']

        edges = np.array(fields.EdgeLengthszWrapper().get_fabs(0,dim,include_ghosts=False)[0])
        edges += np.array(fields.EdgeLengthsxWrapper().get_fabs(0,dim,include_ghosts=False)[0])
        edges += np.array(fields.EdgeLengthsyWrapper().get_fabs(0,dim,include_ghosts=False)[0])
        nx, ny, nz = edges.shape[0], edges.shape[1], edges.shape[2]
        x, y, z = np.arange(nx), np.arange(ny), np.arange(nz)

        fig, (ax1,ax2) = plt.subplots(1,2, tight_layout=True, width_ratios=[int(nz/ny), int(nx/ny)])

        Y, Z = np.meshgrid(y,z)
        cm = ax1.contourf(Z, Y, np.transpose(edges[int(nx/2), :, :]), levels=2)
        ax1.set_title('YZ plane at x=0'),
        ax1.set_xlabel(f'z [ncell]')
        ax1.set_ylabel(f'y [ncell]')   

        X, Y = np.meshgrid(x,y)
        cm = ax2.contourf(X, Y, np.transpose(edges[ :, :, int(nz/2)]), levels=2)
        ax2.set_title('XY plane at z=0'),
        ax2.set_xlabel(f'x [ncell]')
        ax2.set_ylabel(f'y [ncell]') 

        plt.show()

        if save:
            fig.savefig(f'contourEB{label[dim]}.{ext}')

    def testInj(self, nplots=5, save=False, ext='png', return_=False):

        steps0 = int(self.t_inj/self.dt + 2*self.sigmaz/(self.beam_beta*self.c)/self.dt)
        steps1 = int((self.zmax-self.zmin + 2*self.sigmaz)/(self.beam_beta*self.c)/self.dt/nplots)

        self.sim.step(steps0)
        Ez = np.array(fields.EzWrapper().get_fabs(0,2,include_ghosts=False)[0])
        xf, yf, zf = np.arange(Ez.shape[0]), np.arange(Ez.shape[1]), np.arange(Ez.shape[2])

        fig, axs = plt.subplots(nplots, 1, tight_layout=True, figsize=(6,10)) 
        Ez, rho = {}, {}

        for i in range(nplots):
            Ez[i] = np.array(fields.EzWrapper().get_fabs(0,2,include_ghosts=False)[0][self.ixsource, self.iysource, :])
            rho[i] = np.array(fields.JzWrapper().get_fabs(0,2,include_ghosts=False)[0][self.ixsource, self.iysource, :]/(self.beam_beta*self.c))  #[C/m3]
            
            norm_E = np.max(np.abs(Ez[i]))
            norm_r = np.max(rho[i])
            
            axs[i].plot(zf, Ez[i]/norm_E, c='g', lw=1.5, label='$E_z$(0,0,z)')
            axs[i].plot(zf, rho[i]/norm_r, c='r', lw=1.5, label=r'$\rho$(0,0,z)')
            axs[i].set_title(f'Timestep : {steps0+steps1*i}')
            axs[i].set_xlabel(f'z [ncell]')
            axs[i].set_ylabel(f'Ez [V/m]')
            axs[i].legend()

            self.sim.step(steps1)

        plt.show()
        if save:
            fig.savefig(f'injection.{ext}')
        if return_:
            return Ez, rho


