''' Solver module for wake and impedance computation

Wakefields are generated inside the accelerator vacuum chamber
due to interaction of the structure with a passing beam. Among the
properties that characterize their impact on the machine are the beam 
coupling Impedance in frequency domain, and the wake potential in 
time domain. An accurate evaluation of these properties is crucial to 
effectively predict thedissipated power and beam stability. 

Wakis integrates the electromagnetic (EM) wakefields for general 3d 
structures and computes the Wake potential and Impedance for 
longitudinal and transverse planes.

@date: Created on 01.11.2022
@author: Elena de la Fuente

'''

import numpy as np
from tqdm import tqdm

class Solver():
    '''Mixin class to encapsulate solver methods
    '''

    def calc_long_WP(self, Ezt=None, **kwargs):
        '''
        Obtains the wake potential from the pre-computed longitudinal
        Ez(z,t) field from the specified solver. 
        Parameters can be passed as **kwargs.

        Parameters
        ----------
        t : ndarray
            vector containing time values [s]
        z : ndarray
            vector containint z-coordinates [m]
        sigmaz : float
            Beam longitudinal sigma, to calculate injection time
        q : float
            Beam charge, to normalize wake potential
        ti : float, default 8.53*sigmaz/c
            Injection time needed to set the negative part of s vector
            and wakelength
        Ezt : ndarray, default None
            Matrix (nz x nt) containing Ez(x_test, y_test, z, t)
            where nz = len(z), nt = len(t)
        Ez_file : str, default 'Ez.h5'
            HDF5 file containing the Ez(xsource, ysource, z) field data
            for every timestep 
        '''
        for key, val in kwargs.items():
            setattr(self, key, val)

        # Read h5
        if self.Ez_hf is None:
            self.read_Ez()

        if Ezt is not None:
            self.Ezt = Ezt

        # Aux variables
        nt = len(self.t)
        dt = self.t[2]-self.t[1]

        # Injection time
        if self.ti is not None:
            ti = self.ti

        else:
            ti = 8.548921333333334*self.sigmaz/self.c  #injection time as in CST
            self.ti = ti

        if self.zf is None: self.zf = self.z

        nz = len(self.zf)
        dz = self.zf[2]-self.zf[1]
        zmax = np.max(self.zf)
        zmin = np.min(self.zf)               

        # Set Wake length and s
        WL = nt*dt*self.c - (zmax-zmin) - ti*self.c
        s = np.arange(-self.ti*self.c, WL, dt*self.c) 

        self.log.debug('Max simulated time = '+str(round(self.t[-1]*1.0e9,4))+' ns')
        self.log.debug('Wakelength = '+str(round(WL,3))+'m')

        # Initialize 
        WP = np.zeros_like(s)
        keys = list(self.Ez_hf.keys())

        # Assembly Ez field
        if self.Ezt is None:
            self.log.info('Assembling Ez field...')
            Ezt = np.zeros((nz,nt))     #Assembly Ez field
            for n in range(nt):
                Ez = self.Ez_hf[keys[n]]
                Ezt[:, n] = Ez[Ez.shape[0]//2+1,Ez.shape[1]//2+1,:]

            self.Ezt = Ezt

        # integral of (Ez(xtest, ytest, z, t=(s+z)/c))dz
        self.log.info('Calculating longitudinal wake potential WP(s)...')
        with tqdm(total=len(s)*len(self.zf)) as pbar:
            for n in range(len(s)):    
                for k in range(nz): 
                    ts = (self.zf[k]+s[n])/self.c-zmin/self.c-self.t[0]+ti
                    it = int(ts/dt)                 #find index for t
                    WP[n] = WP[n]+(Ezt[k, it])*dz   #compute integral
                    pbar.update(1)

        WP = WP/(self.q*1e12)     # [V/pC]

        self.s = s
        self.WP = WP
        self.wakelength = WL
        

    def calc_long_WP_3d(self, **kwargs):
        '''
        Obtains the 3d wake potential from the pre-computed Ez(x,y,z) 
        field from the specified solver. The calculation 
        Parameters can be passed as **kwargs.

        Parameters
        ----------
        Ez_file : str, default 'Ez.h5'
            HDF5 file containing the Ez(x,y,z) field data for every timestep
        t : ndarray
            vector containing time values [s]
        z : ndarray
            vector containing z-coordinates [m]
        q : float
            Total beam charge in [C]. Default is 1e9 C
        n_transverse_cells : int, default 1
            Number of transverse cells used for the 3d calculation: 2*n+1 
            This determines de size of the 3d wake potential 
        '''
        for key, val in kwargs.items():
            setattr(self, key, val)

        # Read h5
        if self.Ez_hf is None:
            self.read_Ez()

        # Init time
        if self.ti is not None:
            ti = self.ti

        else:
            ti = 8.548921333333334*self.sigmaz/self.c  #injection time as in CST

        # Aux variables
        nt = len(self.t)
        dt = self.t[2]-self.t[1]

        # Longitudinal dimension
        if self.zf is None: self.zf = self.z
        nz = len(self.zf)
        dz = self.zf[2]-self.zf[1]
        zmax = max(self.zf)
        zmin = min(self.zf)               

        # Set Wake length and s
        WL = nt*dt*self.c - (zmax-zmin) - ti*self.c
        s = np.arange(-self.ti*self.c, WL, dt*self.c) 

        self.log.debug('Max simulated time = '+str(round(self.t[-1]*1.0e9,4))+' ns')
        self.log.debug('Wakelength = '+str(round(WL/self.unit_m,0))+' mm')

        #field subvolume in No.cells for x, y
        i0, j0 = self.n_transverse_cells, self.n_transverse_cells    
        WP = np.zeros_like(s)
        WP_3d = np.zeros((i0*2+1,j0*2+1,len(s)))
        keys = list(self.Ez_hf.keys())

        self.log.info('Calculating longitudinal wake potential WP(s)')
        with tqdm(total=len(s)*(i0*2+1)*(j0*2+1)) as pbar:
            for i in range(-i0,i0+1,1):  
                for j in range(-j0,j0+1,1):

                    # Assembly Ez field
                    for n in range(nt):
                        Ez = self.Ez_hf[keys[n]]
                        Ezt[:, n] = Ez[Ez.shape[0]//2+1,Ez.shape[1]//2+1,:]

                    # integral of (Ez(xtest, ytest, z, t=(s+z)/c))dz
                    for n in range(len(s)):    
                        for k in range(0, nz): 
                            ts = (self.zf[k]+s[n])/self.c-zmin/self.c-self.t[0]+ti
                            it = int(ts/dt)                 #find index for t
                            WP[n] = WP[n]+(Ezt[k, it])*dz   #compute integral
                        
                        pbar.update(1)

                    WP = WP/(self.q*1e12)     # [V/pC]
                    WP_3d[i0+i,j0+j,:] = WP 

        self.s = s
        self.WP = WP_3d[i0,j0,:]
        self.WP_3d = WP_3d

    def calc_trans_WP(self, **kwargs):
        '''
        Obtains the transverse wake potential from the longitudinal 
        wake potential in 3d using the Panofsky-Wenzel theorem using a
        second-order scheme for the gradient calculation

        Parameters
        ----------
        WP_3d : ndarray
            Longitudinal wake potential in 3d WP(x,y,s). Shape = (2*n+1, 2*n+1, len(s))
            where n = n_transverse_cells and s the wakelength array
        s : ndarray
            Wakelegth vector s=c*t-z representing the distance between 
            the source and the integration point. Goes from -8.53*sigmat to WL
            where sigmat = sigmaz/c and WL is the Wakelength
        dx : float 
            Ez field mesh step in transverse plane, x-dir [m]
        dy : float 
            Ez field mesh step in transverse plane, y-dir [m]
        x : ndarray, optional
            vector containing x-coordinates [m]
        y : ndarray, optional
            vector containing y-coordinates [m]
        n_transverse_cells : int, default 1
            Number of transverse cells used for the 3d calculation: 2*n+1 
            This determines de size of the 3d wake potential 
        '''

        for key, val in kwargs.items():
            setattr(self, key, val)

        # Obtain dx, dy, ds
        if 'dx' in kwargs.keys() and 'dy' in kwargs.keys(): 
            dx = kwargs['dx']
            dy = kwargs['dy']
        else:
            dx=self.xf[2]-self.xf[1]
            dy=self.yf[2]-self.yf[1]

        ds = self.s[2]-self.s[1]
        i0, j0 = self.n_transverse_cells, self.n_transverse_cells

        # Initialize variables
        WPx = np.zeros_like(self.s)
        WPy = np.zeros_like(self.s)
        int_WP = np.zeros_like(self.WP_3d)

        self.log.info('Calculating transverse wake potential WPx, WPy...')
        # Obtain the transverse wake potential 
        for n in range(len(self.s)):
            for i in range(-i0,i0+1,1):
                for j in range(-j0,j0+1,1):
                    # Perform the integral
                    int_WP[i0+i,j0+j,n]=np.sum(self.WP_3d[i0+i,j0+j,0:n])*ds 

            # Perform the gradient (second order scheme)
            WPx[n] = - (int_WP[i0+1,j0,n]-int_WP[i0-1,j0,n])/(2*dx)
            WPy[n] = - (int_WP[i0,j0+1,n]-int_WP[i0,j0-1,n])/(2*dy)

        self.WPx = WPx
        self.WPy = WPy

    def calc_long_Z(self, samples=1001, **kwargs):
        '''
        Obtains the longitudinal impedance from the longitudinal 
        wake potential and the beam charge distribution using a 
        single-sided DFT with 1000 samples.
        Parameters can be passed as **kwargs

        Parameters
        ----------
        WP : ndarray
            Longitudinal wake potential WP(s)
        s : ndarray
            Wakelegth vector s=c*t-z representing the distance between 
            the source and the integration point. Goes from -8.53*sigmat to WL
            where sigmat = sigmaz/c and WL is the Wakelength
        lambdas : ndarray 
            Charge distribution λ(s) interpolated to s axis, normalized by the beam charge
        chargedist : ndarray, optional
            Charge distribution λ(z). Not needed if lambdas is specified
        q : float, optional
            Total beam charge in [C]. Not needed if lambdas is specified
        z : ndarray
            vector containing z-coordinates [m]. Not needed if lambdas is specified
        sigmaz : float
            Beam sigma in the longitudinal direction [m]. 
            Used to calculate maximum frequency of interest fmax=c/(3*sigmaz)
        '''

        for key, val in kwargs.items():
            setattr(self, key, val)

        self.log.info('Obtaining longitudinal impedance Z...')

        # setup charge distribution in s
        if self.lambdas is None and self.chargedist is not None:
            self.calc_lambdas()
        elif self.lambdas is None and self.chargedist is None:
            self.calc_lambdas_analytic()
            self.log.warning('Using analytic charge distribution λ(s) since no data was provided')

        # Set up the DFT computation
        ds = np.mean(self.s[1:]-self.s[:-1])
        fmax=1*self.c/self.sigmaz/3   #max frequency of interest #TODO: use pi instead of 3
        N=int((self.c/ds)//fmax*samples) #to obtain a 1000 sample single-sided DFT

        # Obtain DFTs
        lambdafft = np.fft.fft(self.lambdas*self.c, n=N)
        WPfft = np.fft.fft(self.WP*1e12, n=N)
        ffft=np.fft.fftfreq(len(WPfft), ds/self.c)

        # Mask invalid frequencies
        mask  = np.logical_and(ffft >= 0 , ffft < fmax)
        WPf = WPfft[mask]*ds
        lambdaf = lambdafft[mask]*ds
        self.f = ffft[mask]            # Positive frequencies

        # Compute the impedance
        self.Z = - WPf / lambdaf
        self.lambdaf = lambdaf

    def calc_trans_Z(self, samples=1001):
        '''
        Obtains the transverse impedance from the transverse 
        wake potential and the beam charge distribution using a 
        single-sided DFT with 1000 samples
        Parameters can be passed as **kwargs
        '''

        self.log.info('Obtaining transverse impedance Zx, Zy...')

        # Set up the DFT computation
        ds = self.s[2]-self.s[1]
        fmax=1*self.c/self.sigmaz/3
        N=int((self.c/ds)//fmax*samples) #to obtain a 1000 sample single-sided DFT

        # Obtain DFTs

        # Normalized charge distribution λ(w) 
        lambdafft = np.fft.fft(self.lambdas*self.c, n=N)
        ffft=np.fft.fftfreq(len(lambdafft), ds/self.c)
        mask  = np.logical_and(ffft >= 0 , ffft < fmax)
        lambdaf = lambdafft[mask]*ds

        # Horizontal impedance Zx⊥(w)
        WPxfft = np.fft.fft(self.WPx*1e12, n=N)
        WPxf = WPxfft[mask]*ds

        self.Zx = 1j * WPxf / lambdaf

        # Vertical impedance Zy⊥(w)
        WPyfft = np.fft.fft(self.WPy*1e12, n=N)
        WPyf = WPyfft[mask]*ds

        self.Zy = 1j * WPyf / lambdaf

    def calc_lambdas(self, **kwargs):
        '''Obtains normalized charge distribution in terms of s 
        λ(s) to use in the Impedance calculation

        Parameters
        ----------
        s : ndarray
            Wakelegth vector s=c*t-z representing the distance between 
            the source and the integration point. Goes from -8.53*sigmat to WL
            where sigmat = sigmaz/c and WL is the Wakelength
        chargedist : ndarray, optional
            Charge distribution λ(z)
        q : float, optional
            Total beam charge in [C]
        z : ndarray
            vector containing z-coordinates of the domain [m]
        zf : ndarray
            vector containing z-coordinates of the field monitor [m]
        '''
        for key, val in kwargs.items():
            setattr(self, key, val)

        if len(self.z) == len(self.chargedist): 
            z = self.z
        elif len(self.zf) == len(self.chargedist):
            z = self.zf
        else: 
            self.log.warning('Dimension error: check inputs dimensions')

        self.lambdas = np.interp(self.s, z, self.chargedist/self.q)

    def calc_lambdas_analytic(self, **kwargs):
        '''Obtains normalized charge distribution in s λ(z)
        as an analytical gaussian centered in s=0 and std
        equal sigmaz
        
        Parameters
        ----------
        s : ndarray
            Wakelegth vector s=c*t-z representing the distance between 
            the source and the integration point. Goes from -8.53*sigmat to WL
            where sigmat = sigmaz/c and WL is the Wakelength
        sigmaz : float
            Beam sigma in the longitudinal direction [m]
        '''

        for key, val in kwargs.items():
            setattr(self, key, val)

        self.lambdas = 1/(self.sigmaz*np.sqrt(2*np.pi))*np.exp(-(self.s**2)/(2*self.sigmaz**2))



    



