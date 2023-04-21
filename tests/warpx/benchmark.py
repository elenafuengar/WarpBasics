import sys
sys.path.append('../../')

import os
import numpy as np
import matplotlib.pyplot as plt
import wakis
import pickle as pk 
import h5py
from scipy.constants import c

def read_dict(path, file):

    with open(path+file,'rb') as handle:
        data = pk.loads(handle.read())

    return data

def read_Ez(path, filename='Ez.h5'):
    '''
    Read the Ez h5 file

    Parameters:
    -----------
    - path = cwd [default] path to the Ez.h5 file. The default is the current working directory 
    - filename = 'Ez.h5' [default]. Specify the name of the Ez file
    '''

    hf = h5py.File(path+filename, 'r')
    print('[PROGRESS] Reading the h5 file: '+ path+filename + ' ...')
    print('[INFO] Size of the file: '+str(round((os.path.getsize(path+filename)/10**9),2))+' Gb')

    # get number of datasets
    size_hf=0.0
    dataset=[]
    n_step=[]
    for key in hf.keys():
        size_hf+=1
        dataset.append(key)

    # get size of matrix
    '''
    Ez_0=hf.get(dataset[0])
    shapex=Ez_0.shape[0]  
    shapey=Ez_0.shape[1] 
    shapez=Ez_0.shape[2] 

    print('[INFO] Ez field is stored in a matrix with shape '+str(Ez_0.shape)+' in '+str(int(size_hf))+' datasets')
    '''
    return hf, dataset

path = '/eos/user/e/edelafue/data/cubcav/'
#path = '/mnt/c/Users/elefu/Documents/CERN/CERNBox/data/cubcav/'

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
chargedist = data.get('charge_dist')

# field parameters
t = data.get('t')               #simulated time [s]
z = data.get('z')               #z axis values  [m]      
z0 = data.get('z0')             #full domain length (+pmls) [m]
x=data.get('x')                 #x axis values  [m]    
y=data.get('y')                 #y axis values  [m]   

# Read Ez 3d data [V/m]
hf, dataset = read_Ez(path, filename='Ez_warpx.h5')
hf_cst, dataset_cst = read_Ez(path, filename='Ez.h5')
z_cst = np.array(hf_cst['z'])
z = np.array(hf['z'])
t = np.array(hf['t'])
dz = z[2]-z[1]
dt = t[2]-t[1]

zmax = np.max(z_cst)
zmin = np.min(z_cst)
steps0 = int(t_inj/dt + sigmaz/c/dt)
steps1 = int((zmax-zmin + sigmaz)/c/dt)

timesteps = [steps0+10, steps0+steps1//2, steps0+steps1-20]
timesteps = [457-50, 467, 457+50]
Ez0 = hf.get(dataset[0])
Ez_cst0=hf_cst.get(dataset[0])
shift = int(10*dz/c/dt)+73

for n in timesteps:
    Ez = hf.get(dataset[n+shift])[Ez0.shape[0]//2+1,Ez0.shape[1]//2+1,:]
    Ez_cst = hf_cst.get(dataset[n])[Ez_cst0.shape[0]//2+1,Ez_cst0.shape[1]//2+1,:]
    norm_r = np.max(Ez)/np.max(chargedist[:,n])

    fig, ax = plt.subplots(1, figsize=(7,3), tight_layout=True)
    ax.plot(z*1e3, Ez, c='g', lw=2.0, label='$E_z$(0,0,z) WarpX')
    ax.plot(z_cst*1e3, Ez_cst, c='k', lw=1.5, ls='--', label='$E_z$(0,0,z) CST')
    ax.plot(z0*0.8*1e3, chargedist[:, n]*norm_r*0.5, c='r', lw=1.5, label=r'$\lambda$(0,0,z)')
    ax.set_title(f'Timestep : {n}')
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('Ez [V/m]')
    ax.set_xlim((zmin*1e3+10,zmax*1e3-10))
    ax.set_ylim((-40000,40000))
    ax.legend()
    plt.show()
    fig.savefig(f'fields{n}.png')
    fig.savefig(f'fields{n}.svg')
