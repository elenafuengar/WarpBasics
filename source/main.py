'''
main.py
===

Procedural version of Wakis
Script to obtain wake potential and impedance  
from pre computed fields

How to use:
---
1. Check main.py and helpers.py are in the same folder
2. Define case: 'cst' or 'warpx'
3. Define path to the needed input files in each case
4. Define beam parameters and integration path if needed

Run with:

 ```
 ipython 
 run wakis.py
 ```

Output
---
wakis.out: binary file with dict
    Contains the ouput of the calculation, stored with pickle

'''
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from scipy.constants import c 
import scipy.interpolate as spi 
import pickle as pk
import h5py as h5py

from helpers import *

cwd = os.getcwd() + '/'

#-----------------------------------------------------------------------------------#

print('---------------------')
print('|  Running WAKIS   |')
print('---------------------')

#---------------------------#
#       Fill variables      #
#---------------------------#

# define EM solver: 'warpx' or 'cst'
case = 'cst'  

if case == 'cst':

    path = '../examples/cst/reswall/'
    # set unit conversion [*m] to [m]
    unit = 1e-3 

    # Beam parameters
    sigmaz = 15*unit                  #beam longitudinal sigma
    q = 1e-9                          #beam charge in [C]
    xsource, ysource = 0e-3, 0e-3     #beam center offset
    xtest, ytest = 0e-3, 0e-3         #integration path offset
    t_inj = sigmaz/c*8.548921333333334    #injection time [s]

    # preprocess if needed
    if not os.path.exists(path+'Ez.h5'):
        try: read_cst_3d(path)
        except: raise Exception('CST 3d field data missing')

    # Read 1d data in dict
    data = read_dict(path, 'cst.inp')

    t = data.get('t')               #simulated time [s]
    z = data.get('z')*unit          #z axis values  [m]      
    x = data.get('x')*unit            #x axis values  [m]    
    y = data.get('y')*unit            #y axis values  [m]   

    # get charge dist [C/m]
    d = read_cst_1d(path, 'lambda.txt')
    charge_dist = np.interp(z, d['d']*unit, d['lambda']) 

    # Read Ez 3d data [V/m]
    hf, dataset = read_Ez(path)

if case == 'warpx':

    path = '/mnt/c/Users/elefu/Documents/CERN/dev/Scripts/WarpX/runs/taper45/'

    data = read_dict(path, 'warpx.inp')
    unit = data.get('unit')

    # beam parameters
    sigmaz = data.get('sigmaz')     #beam longitudinal sigma
    q = data.get('q')               #beam charge in [C]
    t_inj = data.get('init_time')   #injection time [s]

    xsource = data.get('ysource')   #beam center offset
    ysource = data.get('ysource')   #beam center offset
    xtest = data.get('xtest')       #integration path offset
    ytest = data.get('xtest')       #integration path offset

    # charge distribution
    charge_dist = data.get('charge_dist')

    # field parameters
    t = data.get('t')               #simulated time [s]
    z = data.get('z')               #z axis values  [m]      
    z0 = data.get('z0')             #full domain length (+pmls) [m]
    x=data.get('x')                 #x axis values  [m]    
    y=data.get('y')                 #y axis values  [m]   

    # Read Ez 3d data [V/m]
    hf, dataset = read_Ez(path)

    # Show Ez(0,0,z) for every t
    # animate_Ez(path)

    # Plot Ez(0,0,0,t)
    plot_Ez(path, t)

#-----------------------#
#     Obtain W||(s)     #
#-----------------------#
t0 = time.time()

# Aux variables
## t
nt = len(t)
dt = t[1]
## z
if data.get('z0') is None: z0 = z
nz = len(z)
dz = z[2]-z[1]
zmax = max(z)
zmin = min(z)
## z interpolated
zi=np.linspace(zmin, zmax, nt)  
dzi=zi[2]-zi[1]                 

# Set Wake length, s
Wake_length=nt*dt*c - (zmax-zmin) - t_inj*c
ns_neg=int(t_inj/dt)            #obtains the length of the negative part of s
ns_pos=int(Wake_length/(dt*c))  #obtains the length of the positive part of s
s=np.linspace(-t_inj*c, 0, ns_neg) #sets the values for negative s
s=np.append(s, np.linspace(0, Wake_length,  ns_pos))

print('[INFO] Max simulated time = '+str(round(t[-1]*1.0e9,4))+' ns')
print('[INFO] Wakelength = '+str(Wake_length/unit)+' mm')

# Initialize 
Ezi = np.zeros((nt,nt))     #interpolated Ez field
ts = np.zeros((nt, len(s))) #result of (z+s)/c for each z, s

WP = np.zeros_like(s)
WP_3d = np.zeros((3,3,len(s)))

i0=1    #center of the array in x
j0=1    #center of the array in y

print('[PROGRESS] Calculating longitudinal wake potential WP...')
for i in range(-i0,i0+1,1):  
    for j in range(-j0,j0+1,1):

        # Interpolate Ez field
        n=0
        for n in range(nt):
            Ez=hf.get(dataset[n])
            Ezi[:, n]=np.interp(zi, z, Ez[Ez.shape[0]//2+i,Ez.shape[1]//2+j,:])                                                        
        n=0
        for n in range(len(s)):    

            #integral of (Ez(xtest, ytest, z, t=(s+z)/c))dz
            k=0
            for k in range(0, nt): 
                ts[k,n]=(zi[k]+s[n])/c-zmin/c-t[0]+t_inj

                if ts[k,n]>0.0:
                    it=int(ts[k,n]/dt)-1           #find index for t
                    WP[n]=WP[n]+(Ezi[k, it])*dzi   #compute integral

        WP=WP/(q*1e12)     # [V/pC]

        WP_3d[i0+i,j0+j,:]=WP 

#-----------------------#
#      Obtain W⊥(s)     #
#-----------------------#

print('[PROGRESS] Calculating transverse wake potential WPx, WPy...')

# Obtain dx, dy 
dx=x[2]-x[1]
dy=y[2]-y[1]

# Initialize variables
i0 = 1 
j0 = 1
ds = s[2]-s[1]
WPx = np.zeros_like(s)
WPy = np.zeros_like(s)
int_WP = np.zeros_like(WP_3d)

# Obtain the transverse wake potential 
for n in range(len(s)):
    for i in range(-i0,i0+1,1):
        for j in range(-j0,j0+1,1):
            # Perform the integral
            int_WP[i0+i,j0+j,n]=np.sum(WP_3d[i0+i,j0+j,0:n])*ds 

    # Perform the gradient (second order scheme)
    WPx[n] = - (int_WP[i0+1,j0,n]-int_WP[i0-1,j0,n])/(2*dx)
    WPy[n] = - (int_WP[i0,j0+1,n]-int_WP[i0,j0-1,n])/(2*dy)

#--------------------------------#
#      Obtain impedance Z||      #
#--------------------------------#

print('[PROGRESS] Obtaining longitudinal impedance Z...')

#Check input
if WP.ndim > 1:
    WP = WP[1,1,:]

# Obtain charge distribution as a function of s, normalized
if charge_dist.ndim > 1:
    timestep=np.argmax(charge_dist[nz//2, :])   #max at cavity center
    qz=np.sum(charge_dist[:,timestep])*dz       #charge along the z axis
    charge_dist_1d = charge_dist[:,timestep]*q/qz   #total charge in the z axis
    lambdas = np.interp(s, z, charge_dist_1d/q)

lambdas = np.interp(s, z, charge_dist/q)

# Set up the DFT computation
ds = s[2]-s[1]
fmax=1*c/sigmaz/3
N=int((c/ds)//fmax*1001) #to obtain a 1000 sample single-sided DFT

# Obtain DFTs
lambdafft = np.fft.fft(lambdas*c, n=N)
WPfft = np.fft.fft(WP*1e12, n=N)
ffft=np.fft.fftfreq(len(WPfft), ds/c)

# Mask invalid frequencies
mask  = np.logical_and(ffft >= 0 , ffft < fmax)
WPf = WPfft[mask]*ds
lambdaf = lambdafft[mask]*ds
f = ffft[mask]            # Positive frequencies

# Compute the impedance
Z = - WPf / lambdaf

#--------------------------------#
#      Obtain impedance Z⊥       #
#--------------------------------#

print('[PROGRESS] Obtaining transverse impedance Zx, Zy...')

# Obtain charge distribution as a function of s, normalized
lambdas = np.interp(s, z, charge_dist/q)

# Set up the DFT computation
ds = s[2]-s[1]
fmax=1*c/sigmaz/3
N=int((c/ds)//fmax*1001) #to obtain a 1000 sample single-sided DFT

# Obtain DFTs

# Normalized charge distribution λ(w) 
lambdafft = np.fft.fft(lambdas*c, n=N)
ffft=np.fft.fftfreq(len(lambdafft), ds/c)
mask  = np.logical_and(ffft >= 0 , ffft < fmax)
lambdaf = lambdafft[mask]*ds

# Horizontal impedance Zx⊥(w)
WPxfft = np.fft.fft(WPx*1e12, n=N)
WPxf = WPxfft[mask]*ds

Zx = 1j * WPxf / lambdaf

# Vertical impedance Zy⊥(w)
WPyfft = np.fft.fft(WPy*1e12, n=N)
WPyf = WPyfft[mask]*ds

Zy = 1j * WPyf / lambdaf


#--------------------------------#

# Calculate elapsed time
t1 = time.time()
totalt = t1-t0
print('Calculation terminated in %ds' %totalt)

# Save results
data = { 'WP' : WP, 
         's' : s,
         #'k_factor' : k_factor,
         'Z' : Z,
         'f' : f,
         'WPx' : WPx,
         'WPy' : WPy,
         'Zx' : Zx,
         'Zy' : Zy,
         'xsource' : xsource,
         'ysource' : ysource,
         'xtest' : xtest,
         'ytest' : ytest,
         'lambdas' : lambdas
        }

# write the dictionary to a txt using pickle module
with open(path + 'wakis.out', 'wb') as handle:
    pk.dump(data, handle)

#-------------------------#
#      Plot results       #
#-------------------------#

# WPz
fig = plt.figure(1, figsize=(8,5), dpi=150, tight_layout=True)
ax=fig.gca()
ax.plot(s/unit, WP, lw=1.5, c='darkorange', label = 'WPz(s) Wakis')
ax.plot(s/unit, lambdas*max(WP)/max(lambdas), c='r', label= '$\lambda$(s)')

if case == 'cst':
    d = read_cst_1d(path, 'WP.txt')
    ax.plot(d['s'], d['WP'], lw=1, c='k', ls='--', label='WPz(s) CST')

ax.set(title='Longitudinal Wake potential $W_{||}$(s)',
        xlabel='s [mm]',
        ylabel='$W_{||}$(s) [V/pC]',
        ylim=(min(WP)*1.2, max(WP)*1.2) )
ax.legend(loc='upper right')
ax.grid(True, color='gray', linestyle=':')
plt.show()
fig.savefig(path+'WPz.png', bbox_inches='tight')

# Zz
fig = plt.figure(2, figsize=(8,5), dpi=150, tight_layout=True)
ax=fig.gca()
ax.plot(f*1e-9, abs(Z), lw=1.4, c='b', label = 'abs Z(w) Wakis')
ax.plot(f*1e-9, np.real(Z), lw=1.2, c='r', ls='--', label = 'Re Z(w) Wakis')
ax.plot(f*1e-9, np.imag(Z), lw=1.2, c='g', ls='--', label = 'Im Z(w) Wakis')

if case == 'cst':
    d = read_cst_1d(path, 'Z.txt')
    ax.plot(d['f'], d['Z'], lw=1, c='k', ls='--', label='abs Z(w) CST')

ax.set( title='Longitudinal impedance Z||(w)',
        xlabel='f [GHz]',
        ylabel='Z||(w) [$\Omega$]',   
        xlim=(0.,np.max(f)*1e-9)    )
ax.legend(loc='upper left')
ax.grid(True, color='gray', linestyle=':')
plt.show()
fig.savefig(path+'Zz.png', bbox_inches='tight')

# WPx, WPy
fig = plt.figure(3, figsize=(8,5), dpi=150, tight_layout=True)
ax=fig.gca()
ax.plot(s/unit, WPx, lw=1.2, color='g', label='Wx⊥(s)')
ax.plot(s/unit, WPy, lw=1.2, color='magenta', label='Wy⊥(s)')
ax.set(title='Transverse Wake potential W⊥(s) \n (x,y) source = ('+str(round(xsource/unit,1))+','+str(round(ysource/unit,1))+') mm | test = ('+str(round(xtest/unit,1))+','+str(round(ytest/unit,1))+') mm',
        xlabel='s [mm]',
        ylabel='$W_{⊥}$ [V/pC]',
        xlim=(np.min(s/unit), np.max(s/unit)), )
ax.legend(loc='best')
ax.grid(True, color='gray', linestyle=':')
plt.show()
fig.savefig(path+'WPxy.png', bbox_inches='tight')

#Zx, Zy 
fig = plt.figure(3, figsize=(8,5), dpi=150, tight_layout=True)
ax=fig.gca()

ax.plot(f*1e-9, abs(Zx), lw=1.5, c='g', label = 'abs Zx(w) Wakis')
ax.plot(f*1e-9, np.real(Zx), lw=1, c='g', ls=':', label = 'Re Zx(w) Wakis')
ax.plot(f*1e-9, np.imag(Zx), lw=1, c='g', ls='--', label = 'Im Zx(w) Wakis')

ax.plot(f*1e-9, abs(Zy), lw=1.5, c='magenta', label = 'abs Zx(w) Wakis')
ax.plot(f*1e-9, np.real(Zy), lw=1, c='magenta', ls=':', label = 'Re Zy(w) Wakis')
ax.plot(f*1e-9, np.imag(Zy), lw=1, c='magenta', ls='--', label = 'Im Zy(w) Wakis')

ax.set(title='Transverse impedance Z⊥(w) \n (x,y) source = ('+str(round(xsource/unit,1))+','+str(round(ysource/unit,1))+') mm | test = ('+str(round(xtest/unit,1))+','+str(round(ytest/unit,1))+') mm',
        xlabel='f [GHz]',
        ylabel='Z⊥(w) [$\Omega$]',   
        xlim=(0.,np.max(f)*1e-9)   )
ax.legend(loc='best')
ax.grid(True, color='gray', linestyle=':')
plt.show()
fig.savefig(path+'Zxy.png', bbox_inches='tight')