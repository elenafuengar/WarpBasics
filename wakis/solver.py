''' Solver module for wake and impedance computation

Wakefields are generated inside the accelerator vacuum chamber
due to interaction of the structure with a passing beam. Among the
properties that characterize their impact on the machine are the beam 
coupling Impedance in frequency domain, and the wake potential in 
time domain. An accurate evaluation of these properties is crucial to 
effectively predict thedissipated power and beam stability. 

integrates the electromagnetic (EM) wakefields for general 3d 
structures and computes the Wake potential and Impedance for 
longitudinal and transverse planes.

'''

import numpy as np

c = 299792458.0 #[m/s]

class Solver():
    '''Mixin class to encapsulate solver methods
    '''

    def calc_long_WP(self):
        '''
        Obtains the 3d wake potential from the pre-computed Ez field from the 
        specified solver.
        '''

        # Read data
        hf, dataset = self.Ez['hf'], self.Ez['dataset']

        # Aux variables
        nt = len(self.t)
        dt = self.t[-1]/(nt-1)
        ti = 8.53*self.sigmaz/c 

        nz = len(self.z)
        dz = self.z[2]-self.z[1]
        zmax = max(self.z)
        zmin = min(self.z)

        zi = np.linspace(zmin, zmax, nt)  
        dzi = zi[2]-zi[1]                 

        # Set Wake length and s
        WL = nt*dt*c - (zmax-zmin) - ti*c
        ns_neg = int(ti/dt)             #obtains the length of the negative part of s
        ns_pos = int(WL/(dt*c))             #obtains the length of the positive part of s
        s = np.linspace(-ti*c, 0, ns_neg) #sets the values for negative s
        s = np.append(s, np.linspace(0, WL,  ns_pos))

        self.log.debug('Max simulated time = '+str(round(self.t[-1]*1.0e9,4))+' ns')
        self.log.debug('Wakelength = '+str(round(WL/self.unit_m,0))+' mm')

        # Initialize 
        Ezi = np.zeros((nt,nt))     #interpolated Ez field
        ts = np.zeros((nt, len(s))) #result of (z+s)/c for each z, s

        WP = np.zeros_like(s)
        WP_3d = np.zeros((3,3,len(s)))

        #choose different subvolume width [TODO]
        i0, j0 = 1, 1    #center of the subvolume in No.cells for x, y

        self.log.info('Calculating longitudinal wake potential WP')
        for i in range(-i0,i0+1,1):  
            for j in range(-j0,j0+1,1):

                # Interpolate Ez field
                for n in range(nt):
                    Ez = hf.get(dataset[n])
                    Ezi[:, n] = np.interp(zi, self.z, Ez[Ez.shape[0]//2+i,Ez.shape[1]//2+j,:])  

                #integral of (Ez(xtest, ytest, z, t=(s+z)/c))dz
                for n in range(len(s)):    
                    for k in range(0, nt): 
                        ts[k,n] = (zi[k]+s[n])/c-zmin/c-self.t[0]+ti

                        if ts[k,n]>0.0:
                            it = int(ts[k,n]/dt)-1            #find index for t
                            WP[n] = WP[n]+(Ezi[k, it])*dzi    #compute integral

                WP = WP/(self.q*1e12)     # [V/pC]
                WP_3d[i0+i,j0+j,:] = WP 

        self.s = s
        self.WP = WP_3d[i0,j0,:]

        return WP_3d, i0, j0

    def calc_trans_WP(self, WP_3d, i0, j0):
        '''
        Obtains the transverse wake potential from the longitudinal 
        wake potential in 3d using the Panofsky-Wenzel theorem
        '''

        self.log.info('Calculating transverse wake potential WPx, WPy...')

        # Obtain dx, dy, ds
        dx=self.x[2]-self.x[1]
        dy=self.y[2]-self.y[1]
        ds = self.s[2]-self.s[1]

        # Initialize variables
        WPx = np.zeros_like(self.s)
        WPy = np.zeros_like(self.s)
        int_WP = np.zeros_like(WP_3d)

        # Obtain the transverse wake potential 
        for n in range(len(self.s)):
            for i in range(-i0,i0+1,1):
                for j in range(-j0,j0+1,1):
                    # Perform the integral
                    int_WP[i0+i,j0+j,n]=np.sum(WP_3d[i0+i,j0+j,0:n])*ds 

            # Perform the gradient (second order scheme)
            WPx[n] = - (int_WP[i0+1,j0,n]-int_WP[i0-1,j0,n])/(2*dx)
            WPy[n] = - (int_WP[i0,j0+1,n]-int_WP[i0,j0-1,n])/(2*dy)

        self.WPx = WPx
        self.WPy = WPy

    def calc_long_Z(self):
        '''
        Obtains the longitudinal impedance from the longitudinal 
        wake potential and the beam charge distribution using a 
        single-sided DFT with 1000 samples
        '''
        self.log.info('Obtaining longitudinal impedance Z...')

        # setup charge distribution in s
        self.lambdas = np.interp(self.s, self.z, self.chargedist/self.q)

        # Set up the DFT computation
        ds = self.s[2]-self.s[1]
        fmax=1*c/self.sigmaz/3   #max frequency of interest
        N=int((c/ds)//fmax*1001) #to obtain a 1000 sample single-sided DFT

        # Obtain DFTs
        lambdafft = np.fft.fft(self.lambdas*c, n=N)
        WPfft = np.fft.fft(self.WP*1e12, n=N)
        ffft=np.fft.fftfreq(len(WPfft), ds/c)

        # Mask invalid frequencies
        mask  = np.logical_and(ffft >= 0 , ffft < fmax)
        WPf = WPfft[mask]*ds
        lambdaf = lambdafft[mask]*ds
        self.f = ffft[mask]            # Positive frequencies

        # Compute the impedance
        self.Z = - WPf / lambdaf
        self.lambdaf = lambdaf

    def calc_trans_Z(self):
        '''
        Obtains the transverse impedance from the transverse 
        wake potential and the beam charge distribution using a 
        single-sided DFT with 1000 samples
        '''

        self.log.info('Obtaining transverse impedance Zx, Zy...')

        # Set up the DFT computation
        ds = self.s[2]-self.s[1]
        fmax=1*c/self.sigmaz/3
        N=int((c/ds)//fmax*1001) #to obtain a 1000 sample single-sided DFT

        # Obtain DFTs

        # Normalized charge distribution λ(w) 
        lambdafft = np.fft.fft(self.lambdas*c, n=N)
        ffft=np.fft.fftfreq(len(lambdafft), ds/c)
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

