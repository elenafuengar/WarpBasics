'''
Wakis main class to manage attributes and methods 
across the modules

It can be instantiated from a previous output file
or constructed through inputs module classes, from 
which it inherits the attributes.

@date: Created on 20.10.2022
@author: Elena de la Fuente
'''

import os
import time
import json
import pickle as pk 

#dependencies
import numpy as np 
import matplotlib.pyplot as plt

from wakis.logger import get_logger
from wakis.bucket import Bucket

class Wakis(Bucket):
    '''
    Central class to manage attributes and methods across the modules

    Attributes include chosen solver, units, beam parameters, integration  
    path, EM field and charge distribution data, and verbose level
    Methods include setters for atributes and logger initialization
    '''

    def __init__(self, solver=None, input_file=None, q=1e-9, sigmaz=1e-3, 
                 xsource=0., ysource=0., xtest=0., ytest=0., chargedist=None, ti=None,
                 rho=None, Ez_file='Ez.h5', unit_m=1e-3, unit_t=1e-9, unit_f=1e9, 
                 path=None, verbose=2, **kwargs):
        '''
        Parameters
        ----------
        unit_m : str or float
            Dimensional units given as str: 'mm', 'cm', 'dm', or as float: 1e-3 for mm
            Default: 'mm'
        unti_t : str or float
            Time units given as str: 'ns', 'ps', 'ms', 'us', or as float: 1e-9 for ns
            Default: 'ns'
        unit_f: str or float
            Frequency units given as str: 'GHz', 'MHz', 'kHz', or as float: 1e9 for MHz
            Default: 'GHz'
        q : float
            Beam total charge in [C]
        sigmaz : float 
            Beam sigma in the longitudinal direction [m]
        xsource : float
            Beam center in the transverse plane, x-dir [m]
        ysource : float
            Beam center in the transverse plane, y-dir [m]
        xtest : float
            Integration path center in the transverse plane, x-dir [m]
        ytest : float
            Integration path center in the transverse plane, y-dir [m]
        ti : float 
            Injection time, when beam enters domain [s]
        chargedist : dict or str, default 'lambda.txt'
            When str, specifies the filename containing the charge distribution data
            When dict, contains the charge distribution data in keys: {'X','Y'}
            'X' : longitudinal coordinate [m]
            'Y' : charge distribution in [C/m]
        rho : h5py obj or str
            Contains the (z,t) charge distribution. Each timestep stored in a dataset
            When str, specifies the name to the '.h5' file containing the data
        Ez_file : str, default 'Ez.h5'
            hdf5 file containing Ez(x,y,z) data for every timestep
        input_file : str
            pickled or json file containing field coordinates data. 
            Recommended for warpx solver: input_file = 'warpx.dat'
            Recommended for warpx solver: input_file = 'cst.dat'
        input
        '''

        #constants
        self.c = 299792458.0 #[m/s]

        #user
        self.solver = solver  
        self.input_file = input_file
        self.path = path
        self.verbose = verbose
        self.log = get_logger(level=verbose)
        self.units(unit_m=unit_m, unit_f=unit_f, unit_t=unit_t)

        #beam
        self.q = q
        self.sigmaz = sigmaz
        self.xsource, self.ysource = xsource, xtest
        self.xtest, self.ytest = xtest, ytest
        self.chargedist = None
        self.rho = rho
        self.ti = ti

        #field
        self.Ez_file = Ez_file
        self.Ez_hf = None
        self.Ezt = None #Ez(x_t, y_t, z, t)
        self.t = None
        self.xf, self.yf, self.zf = None, None, None    #field subdomain
        self.x, self.y, self.z = None, None, None #full simulation domain

        #solver init
        self.s = None
        self.lambdas = None
        self.WP = None
        self.WP_3d = None
        self.n_transverse_cells = 1
        self.WPx, self.WPy = None, None
        self.f = None
        self.Z = None
        self.Zx, self.Zy = None, None
        self.lambdaf = None

        #plotting 
        self._figsize = (6,4)
        self._dpi = 150

        if path is None:
            self.path = os.getcwd() + '/'

        #hadlde kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    def units(self, unit_m=1e-3, unit_f=1e9, unit_t=1e-9):
        '''Set wakis units for plotting 

        Parameters
        ----------
        unit_m : str or float
            Dimensional units given as str: 'mm', 'cm', 'dm', or as float: 1e-3 for mm
            Default: 'mm'
        unti_t : str or float
            Time units given as str: 'ns', 'ps', 'ms', 'us', or as float: 1e-9 for ns
            Default: 'ns'
        unit_f: str or float
            Frequency units given as str: 'GHz', 'MHz', 'kHz', or as float: 1e9 for MHz
            Default: 'GHz'

        Raises
        ------
        AssertionError
            If the case chosen is not in the available solvers list: 'warpx', 'cst'
        TypeError
            If the input is not a 'str' or a 'float'
        '''

        if isinstance(unit_m, str):
            if unit_m == 'mm': self.unit_m = 1e-3
            if unit_m == 'cm': self.unit_m = 1e-2
            if unit_m == 'dm': self.unit_m = 1e-1

        elif isinstance(unit_m, float): self.unit_m = unit_m
        else: raise TypeError('Non valid dimensional units. Input type must be "str" or "float"')

        if isinstance(unit_t, str):
            if unit_t == 'ns': self.unit_t = 1e-9
            if unit_t == 'ps': self.unit_t = 1e-12
            if unit_t == 'ms': self.unit_t = 1e-3
            if unit_t == 'us': self.unit_t = 1e-6

        elif isinstance(unit_t, float): self.unit_t = unit_t
        else: raise TypeError('Non valid time units. Input type must be "str" or "float"')

        if isinstance(unit_f, str):
            if unit_f == 'GHz': self.unit_f = 1e9
            if unit_f == 'MHz': self.unit_f = 1e6
            if unit_f == 'kHz': self.unit_f = 1e3

        elif isinstance(unit_f, float): self.unit_f = unit_f
        else: raise TypeError('Non valid frequency units. Input type must be "str" or "float"')

    def beam(self, solver=None, q=None, sigmaz=None, xsource=None, ysource=None, 
             xtest=None, ytest=None, chargedist=None, rho=None):
        '''Alternative method to set wakis beam parameters
        These can be set directly through the __init__ of the class
        
        Parameters 
        ----------
        q : float
            Beam total charge in [C]
        sigmaz : float 
            Beam sigma in the longitudinal direction [m]
        xsource : float
            Beam center in the transverse plane, x-dir [m]
        ysource : float
            Beam center in the transverse plane, y-dir [m]
        xtest : float
            Integration path center in the transverse plane, x-dir [m]
        ytest : float
            Integration path center in the transverse plane, y-dir [m]
        chargedist : dict or str
            When str, specifies the filename containing the charge distribution data
            When dict, contains the charge distribution data in keys: {'X','Y'}
            'X' : longitudinal coordinate [m]
            'Y' : charge distribution in [C/m]
        rho : h5py obj or str
            Contains the (z,t) charge distribution. Each timestep stored in a dataset
            When str, specifies the name to the '.h5' file containing the data
        '''
        self.q = q
        self.sigmaz = sigmaz
        self.xsource, self.ysource = xsource, ysource
        self.xtest, self.ytest = xtest, ytest
        self.chargedist = chargedist
        self.rho = rho

        if isinstance(chargedist, str) and solver == 'cst':
            self.chargedist = self.read_cst_1d(self, filename=chargedist)

    def fields(self, Ez_file='Ez.h5', input_file=None, input_folder=None, path=None, 
                t=None, xf=None, yf=None, zf=None, x=None, y=None, z=None):

        '''Set Electric field data. 
        All defaults are None unless specified otherwise.

        Parameters
        ----------
        Ez_file : str, default 'Ez.h5'
            hdf5 file containing Ez(x,y,z) data for every timestep
        input_file : str
            pickled or json file containing field coordinates data. 
            Recommended for warpx solver: input_file = 'warpx.dat'
            Recommended for warpx solver: input_file = 'cst.dat'
        input_folder : str
            Name of the folder containing the Ez .txt files exported from CST 
            Folder should be in the current directory, otherwise specify path
            Recommended for cst solver: input_folder = '3d' 
        path : str
            Absolute path to the input_folder. Default is set to cwd.
        t : ndarray
            vector containing each timestep time value
        xf : ndarray 
            vector containing Ez field x-coordinates
        yf : ndarray
            vector containing Ez field y-coordinates
        zf : ndarray
            vector containing Ez field z-coordinates
        x : ndarray
            vector containing domain x-coordinates
        y : ndarray
            vector containing domain y-coordinates            
        z : ndarray
            vector containing domain z-coordinates
        '''

        if len(Ez_file.split('/')) > 1:
            self.Ez_file = Ez_file.split('/')[-1]
        else:
            self.Ez_file = Ez_file
            
        self.t = t
        self.xf, self.yf, self.zf = xf, yf, zf        #field subdomain
        self.x, self.y, self.z = x, y, z  #full simulation domain

        if input_folder is not None and self.solver == 'cst':
            self.read_cst_3d(path=path, folder=input_folder)
        if input_file is not None and self.solver == 'warpx':
            self.read_warpx(filename=input_file)
        if input_file is not None and self.solver == 'cst':
            self.read_cst(filename=input_file)

        '''
        if self.Ez_hf is None:
            try:
                hf, dataset = self.read_Ez(filename=Ez_file)
                self.Ez_hf = {'hf': hf, 'dataset': dataset}
            except:
                self.log.warning(f'{Ez_file} not found or could not be opened')
        '''

    @classmethod
    def from_inputs(cls, *clss): 
        '''
        Factory method from input's module
        classes: User, Beam and Field
        '''
        d = {}
        for cl in clss:
            d.update(cl.__dict__)

        return cls(**d)

    @classmethod
    def from_file(cls, file='wakis.dat'):
        '''
        Set attributes from wakis output file

        Parameters
        ----------
        file : str, default 'wakis.dat'
            Name of the file to read. Both json and pickle IO 
            are supported
        '''
        exts = ['pk', 'pickle', 'out', 'dat']
        ext = file.split('.')[-1]

        if ext == 'js' or ext == 'json':
            with open(file, 'r') as f:
                d = {k: np.array(v) for k, v in js.loads(f.read()).items()}
            return cls(**d)

        elif ext in exts:
            with open(file, 'rb') as f:
                d = pk.load(f)
            return cls(**d)

        else:
            get_logger(2).warning(f'"{f}" file format not supported')        

    def solve(self):
        '''
        Perform the wake potential and impedance for
        longitudinal and transverse plane and display
        calculation time

        Functions are specified in solver.py
        '''
        t0 = time.time()

        # Obtain longitudinal Wake potential
        self.calc_long_WP_3d()

        #Obtain transverse Wake potential
        self.calc_trans_WP()

        #Obtain the longitudinal impedance
        self.calc_long_Z()

        #Obtain transverse impedance
        self.calc_trans_Z()

        #Elapsed time
        t1 = time.time()
        totalt = t1-t0
        self.log.info('Calculation terminated in %ds' %totalt)

    def save(self, ext = 'dat'):
        '''
        Save results in 'wakis' file. 
        Two dumping methods supported: 'json' and 'pickle'

        Parameters
        ----------
        ext :  str, default 'dat'
            Extention to be used in output file 'json' or 'pickle'

        Raises
        ------
        Warning
            If the file extension is not supported
        '''
        d = self.__dict__
        keys = ['s', 'WP', 'WPx', 'WPy', 'f', 'Z', 'Zx', 'Zy', \
                'lambdas', 'xsource', 'ysource', 'xtest', 'ytest', \
                'q', 'sigmaz', 't', 'x', 'y', 'z', \
                'unit_m', 'unit_f', 'unit_t', 'path', 'Ez_file']

        exts = ['pk', 'pickle', 'out', 'dat']

        if ext == 'json':
            j = json.dumps({k: d[k].tolist() for k in keys})
            with open('wakis.' + ext, 'w') as f:
                json.dump(j, f)
            self.log.info('"wakis.' + ext +'" file succesfully generated') 

        elif ext in exts:
            p = {k: d[k] for k in keys}
            with open('wakis.' + ext, 'wb') as f:
                pk.dump(p, f)
            self.log.info('"wakis.' + ext +'" file succesfully generated') 
        
        else: 
            self.log.warning(f'Extension ".{ext}" not supported')


    def plot(self, Z='abs', Zx='abs', Zy='abs', chargedist=True, return_=True): 
        '''Plot results in different figures that are
        returned as dictionaries

        Parameters
        ----------
        Z : str, default 'abs'
            Content of impedance plot: 
            'abs' : magnitude 
            'Re' : real only
            'Im' : imaginary only
            'all' : abs, real and im
            'ReIm' : real and im
            'none' : no line
        Zx : str, default 'abs'
            Content of transverse impedance plot, x
        Zy : str, default 'abs'
            Content of transverse impedance plot, y
        chargedist : bool, default True
            Plot charge distribution on top of wake potential
        return_ : bool, default False
            Return or not the fig and axis handles

        Returns
        -------
        figs : dict
            Dictionary containing the figure hanldes. Keys:
            '1' : Longitudinal wake potential
            '2' : Longitudinal impedance
            '3' : Transverse wake potentials
            '4' : Transverse impedances
            '5' : Charge distribution
        axs : dict
            Dictionary containing the axes hanldes. Keys:
            '1' : Longitudinal wake potential
            '2' : Longitudinal impedance
            '3' : Transverse wake potentials
            '4' : Transverse impedances
            '5' : Charge distribution
        '''

        figs = {}
        axs = {}

        figs['1'], axs['1'] = self.plot_long_WP(self)
        figs['2'], axs['2'] = self.plot_long_Z(self, plot=Z)
        figs['3'], axs['3'] = self.plot_trans_WP(self)
        figs['4'], axs['4'] = self.plot_trans_Z(self, plot=(Zx, Zy))
        figs['5'], axs['5'] = self.plot_charge_dist(self)

        return figs, axs

    def subplot(self, save=True, return_=True):
        ''' Subplot with all wakis results in the same 
        figure and returns each ax as a dictionary

        Parameters
        ----------
        save : bool, default True
            Flag to enable saving the plotted figure in '.png' format
        return_ : bool, default False
            Controls wether to return figure and axis handles

        Returns
        -------
        fig : figure object
            Matplotlib figure containing the subplot
        axs : tupple
            Tupple containing the axis handles (ax1, ax2), (ax3, ax4)
            '1' : Longitudinal wake potential
            '2' : Longitudinal impedance
            '3' : Transverse wake potentials
            '4' : Transverse impedances
        '''

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.set_size_inches(16, 10)

        plt.text(x=0.5, y=0.96, s="WAKIS wake solver result", 
                fontsize='x-large', fontweight='bold', ha="center", transform=fig.transFigure)
        plt.text(x=0.5, y=0.93, s= '(x,y) source = ('+str(round(self.xsource/1e3,1))+','+str(round(self.ysource/1e3,1))+') mm | test = ('+str(round(self.xtest/1e3,1))+','+str(round(self.ytest/1e3,1))+') mm', 
                fontsize='large', ha="center", transform=fig.transFigure)  

        f, ax1 = self.plot_long_WP(self, fig = fig, ax = ax1, chargedist = True)
        f, ax2 = self.plot_long_Z(self, fig = fig, ax = ax2, plot='all')
        f, ax3 = self.plot_trans_WP(self, fig = fig, ax = ax3)
        fig, ax4 = self.plot_trans_Z(self, fig = fig, ax = ax4,plot='all')

        plt.show()

        if save: fig.savefig(self.path+'wakis.png')

        if return_:
            return fig, ((ax1, ax2), (ax3, ax4))

    def __str__(self):
        return f'Wakis atributes: \n' + \
        '- beam: \n q={self.q}, sigmaz={self.sigma}, xsource={self.xsource}, ysource={self.ysource}, xtest={self.xtest}, ytest={self.ytest} \n' +\
        '- field: \n Ez={self.Ez}, \n t={self.t}, \n x={self.xf}, y={self.y}, z={self.z}, x0={self.x}, y0={self.y0}, z0={self.z0} \n' + \
        '- charge distribution: \n chargedist={self.chargedist} \n'
        '- solver: \n s={s}, lambdas={self.lambdas}, WP={self.WP}, Z={self.Z}, WPx={self.WPx}, WPy={self.WPy}, Zx={self.Zx}, Zy={self.Zy} \n'
