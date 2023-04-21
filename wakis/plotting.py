'''Plotting module for Wakis inputs / outputs

Includes functions to plot wakis computed wake 
potential and impedance for longitudinal and 
transverse planes, and funtions to display the
electric field in animated plot or contour for 
each timestep

@date: Created on 24.10.2022
@author: Elena de la Fuente
'''

import matplotlib.pyplot as plt
import numpy as np

class Plotter():
    '''Mixin class to encapsulate plotting methods
    '''
    
    def animate_Ez(self, flag_chargedist = True):
        '''
        Creates an animated plot showing the Ez(0,0,z) field plot for every timestep

        Parameters:
        -----------
        flag_chargedist : :onj: `bool`, optional
            Flag to plot the charge distribution on top of the electric field evolution
            Only supported if self.rho is not None
        '''

        # Read data
        hf, dataset = self.Ez_hf['hf'], self.Ez_hf['dataset']

        # Extract field on axis Ez(0,0,z,t)
        Ez0=[]
        for n in range(len(dataset)):
            Ez=hf.get(dataset[n]) # [V/m]
            Ez0.append(np.array(Ez[Ez.shape[0]//2, Ez.shape[1]//2,:])) # [V/m]

        Ez0=np.transpose(np.array(Ez0))


        plt.ion()
        n=0
        for n in range(10,1000):
            if n % 1 == 0:
                #--- Plot Ez along z axis 
                fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
                ax=fig.gca()

                if self.rho is not None and flag_chargedist:
                    ax.plot(self.z, self.rho[:,n]/np.max(self.rho)*np.max(Ez0)*0.4, lw=1.3, color='r', label='$\lambda $') 
                
                ax.plot(self.z, Ez0[:, n], color='g', label='Ez(0,0,z)')
                ax.set(title='Electric field at time = '+str(round(self.t[n]*1e9,2))+' ns | timestep '+str(n),
                        xlabel='z [m]',
                        ylabel='E [V/m]',         
                        ylim=(-np.max(Ez0)*1.1,np.max(Ez0)*1.1),
                        xlim=(min(self.z),max(self.z)),
                                )
                ax.legend(loc='best')
                ax.grid(True, color='gray', linestyle=':')
                fig.canvas.draw()
                fig.canvas.flush_events()
                fig.clf()

        plt.close()


    def contour_Ez(self, vmin=-1.0e5, vmax=1.0e5):
        '''
        Creates an animated contour of the Ez field in the Y-Z plane at x=0

        Parameters:
        -----------
        vmin : float
            Minimum value of the colorbar. Default -1.0e5
        vmax : float
            Maximum value of the colorbar. Default +1.0e5
        '''

        # Read data
        hf, dataset = self.Ez_hf['hf'], self.Ez_hf['dataset']
        if self.y0 is not None : y = self.y0
        else: y = self.y

        plt.ion()
        for n in range(len(dataset)):
            Ez=hf.get(dataset[n])
            fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
            ax=fig.gca()
            im=ax.imshow(Ez[int(Ez.shape[0]/2),:,:], vmin=vmin, vmax=vmax, extent=[min(self.z), max(self.z), min(y), max(y)], cmap='jet')
            ax.set(title='WarpX Ez field, t = ' + str(round(self.t[n]*1e9,3)) + ' ns',
                   xlabel='z [mm]',
                   ylabel='y [mm]'
                   )
            plt.colorbar(im, label = 'Ez [V/m]')
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            fig1.clf() 

        plt.close()


    def plot_charge_dist(self, fig=None, ax=None):
        '''
        Plots the charge distribution λ(s) 
        '''

        show = False
        if fig is None or ax is None: 
            fig = plt.figure(5, figsize=self._figsize, dpi=self._dpi, tight_layout=True)
            ax=fig.gca()
            show = True

        ax.plot(self.s, self.lambdas, lw=1.2, color='red', label='$\lambda$(s)')
        ax.set(title='Charge distribution $\lambda$(s)',
                xlabel='s [m]',
                ylabel='$\lambda$(s) [C/m]',
                xlim=(min(self.s), np.max(self.s))
                )
        ax.legend(loc='best')
        ax.grid(True, color='gray', linestyle=':')
         
        if show: plt.show()

        return fig, ax

    def plot_long_WP(self, fig=None, ax=None, compare=False, chargedist=False, units=1e-3):
        '''
        Plots the longitudinal wake potential W||(s) 
        '''

        show = False
        if fig is None or ax is None:
            fig = plt.figure(1, figsize=self._figsize, dpi=self._dpi, tight_layout=True)
            ax = fig.gca()
            show = True

        if chargedist:
            ax.plot(self.s, self.lambdas, lw=1.2, color='red', label='$\lambda$(s)')

        ax.plot(self.s, self.WP, lw=1.5, color='darkorange', label='WPz(s)')
        
        if compare:
            d = self.read_cst_1d('WP.txt')
            ax.plot(d['X']*units, d['Y'], lw=1., c='k', ls='--', label='WPz(s) CST')

        ax.set(title='Longitudinal Wake potential $W_{||}$(s)',
                xlabel='s [m]',
                ylabel='$W_{||}$(s) [V/pC]',
                xlim=(min(self.s),np.max(self.s)),
                ylim=(min(self.WP)*1.2, max(self.WP)*1.2)
                )
        ax.legend(loc='lower right')
        ax.grid(True, color='gray', linestyle=':')
         
        if show: plt.show()

        return fig, ax

    def plot_long_Z(self, fig=None, ax=None, plot = 'abs'):
        '''
        Plots the longitudinal impedance Z||(w)
        
        Parameters
        ----------
        plot : :obj: `str`, optional
            Set which impedance value to plot: 'abs', 'real, 'imag', 'all'  
        '''

        if np.iscomplex(self.Z[1]):
            ReZ=np.real(self.Z)
            ImZ=np.imag(self.Z)
            Z=abs(self.Z)
        else: Z = self.Z

        show = False
        if fig is None or ax is None:
            fig = plt.figure(2, figsize=self._figsize, dpi=self._dpi, tight_layout=True)
            ax=fig.gca()
            show = True

        if plot == 'real' or plot == 'all':
            ax.plot(self.f*1e-9, ReZ, lw=1, color='r', markersize=2., label='Real Z(f)')

        if plot == 'imag' or plot == 'all':
            ax.plot(self.f*1e-9, ImZ, lw=1, color='g', markersize=2., label='Imag Z(f)')

        if plot == 'abs' or plot == 'all':
            ifmax=np.argmax(Z)
            ax.plot(self.f[ifmax]*1e-9, Z[ifmax], marker='o', markersize=4.0, color='blue')
            ax.annotate(str(round(self.f[ifmax]*1e-9,2))+ ' GHz', xy=(self.f[ifmax]*1e-9,Z[ifmax]*0.8), xytext=(-20,1), textcoords='offset points', color='blue') 
            ax.plot(self.f*1e-9, Z, lw=1.2, color='b', label='Z||(f) magnitude')
        
        ax.set( title='Longitudinal impedance Z||(f)',
                xlabel='f [GHz]',
                ylabel='Z||(f) [$\Omega$]',   
                xlim=(0.,np.max(self.f)*1e-9)      
                )
        ax.legend(loc='upper left')
        ax.grid(True, color='gray', linestyle=':')
         
        if show: plt.show()

        return fig, ax

    def plot_trans_WP(self, fig=None, ax=None):
        '''
        Plots the transverse wake potential Wx⊥(s), Wy⊥(s) 
        '''

        show = False
        if fig is None or ax is None:
            fig = plt.figure(3, figsize=self._figsize, dpi=self._dpi, tight_layout=True)
            ax=fig.gca()
            show = True

        ax.plot(self.s, self.WPx, lw=1.2, color='g', label='WPx⊥(s)')
        ax.plot(self.s, self.WPy, lw=1.2, color='magenta', label='WPy⊥(s)')
        ax.set(title='Transverse Wake potential W⊥(s) \n (x,y) source = ('+str(round(self.xsource/1e-3,1))+','+str(round(self.ysource/1e-3,1))+') mm | test = ('+str(round(self.xtest/1e-3,1))+','+str(round(self.ytest/1e-3,1))+') mm',
                xlabel='s [m]',
                ylabel='$W_{⊥}$ [V/pC]',
                xlim=(np.min(self.s), np.max(self.s)),
                )
        ax.legend(loc='best')
        ax.grid(True, color='gray', linestyle=':')
         
        if show: plt.show()

        return fig, ax

    def plot_trans_Z(self, fig=None, ax=None, plot = 'abs'):
        '''
        Plots the transverse impedance Zx⊥(w), Zy⊥(w) 

        Parameters
        ----------
        plot : :obj: `str`, optional
            Set which impedance value to plot: 'abs', 'real, 'imag', 'all'  
        '''

        if np.iscomplex(self.Zx[1]):
            ReZx=np.real(self.Zx)
            ImZx=np.imag(self.Zx)
            Zx=abs(self.Zx)

        if np.iscomplex(self.Zy[1]):
            ReZy=np.real(self.Zy)
            ImZy=np.imag(self.Zy)
            Zy=abs(self.Zy)

        show = False
        if fig is None or ax is None:
            fig = plt.figure(4, figsize=self._figsize, dpi=self._dpi, tight_layout=True)
            ax=fig.gca()
            show = True

        #--- plot Zx⊥(w)
        if plot == 'real' or plot == 'all':
            ax.plot(self.f*1e-9, ReZx, lw=1, color='green', ls = '--', label='Real Zx⊥(f)')

        if plot == 'imag' or plot == 'all':
            ax.plot(self.f*1e-9, ImZx, lw=1, color='green' ,ls = ':', label='Imag Zx⊥(f)')

        if plot == 'abs' or plot == 'all':
            ifxmax=np.argmax(Zx)
            ax.plot(self.f[ifxmax]*1e-9, Zx[ifxmax], color='green', marker='o', markersize=4.0)
            ax.annotate(str(round(self.f[ifxmax]*1e-9,2))+ ' GHz', xy=(self.f[ifxmax]*1e-9,Zx[ifxmax]), xytext=(-50,-5), textcoords='offset points', color='green') 
            ax.plot(self.f*1e-9, Zx, lw=1.2, color='green', label='Zx⊥(w)')

        #--- plot Zy⊥(w)
        if plot == 'real' or plot == 'all':
            ax.plot(self.f*1e-9, ReZy, lw=1, color='magenta', ls = '--', label='Real Zy⊥(f)')

        if plot == 'imag' or plot == 'all':
            ax.plot(self.f*1e-9, ImZy, lw=1, color='magenta', ls = ':', label='Imag Zy⊥(f)')

        if plot == 'abs' or plot == 'all':
            ifymax=np.argmax(Zy)
            ax.plot(self.f[ifymax]*1e-9, Zy[ifymax], marker='o', markersize=4.0, color='red')
            ax.annotate(str(round(self.f[ifymax]*1e-9,2))+ ' GHz', xy=(self.f[ifymax]*1e-9, Zy[ifymax]), xytext=(-50,-5), textcoords='offset points', color='magenta') 
            ax.plot(self.f*1e-9, Zy, lw=1.2, color='magenta', label='Zy⊥(f)')

        ax.set(title='Transverse impedance Z⊥(f) \n (x,y) source = ('+str(round(self.xsource/1e-3,1))+','+str(round(self.ysource/1e-3,1))+') mm | test = ('+str(round(self.xtest/1e-3,1))+','+str(round(self.ytest/1e-3,1))+') mm',
                xlabel='f [GHz]',
                ylabel='Z⊥(f) [$\Omega$]',   
                xlim=(0.,np.max(self.f)*1e-9)      
                )

        ax.legend(loc='best')
        ax.grid(True, color='gray', linestyle=':')
         
        if show: plt.show()

        return fig, ax

