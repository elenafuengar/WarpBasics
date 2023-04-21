'''
helpers.py 
===

Helper functions for Wakis main routine

Contents 
---

* Functions to pre-process and read data from EM solvers
    * For CST: read 1d for ASCII files, pre-process field monitod 3d data
    * For WarpX: read dict for input info, read Ez for .h5 field data
    * For PBCI: read 1d ASCII files

* Functions to plot results 
    * Plot Ez(t) at a given point (x,y,z) 
    * Animate Ez(0,0,z,t) 
    
'''


import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import glob
from scipy.constants import c 
import scipy.interpolate as spi 
import pickle as pk
import h5py as h5py


def read_dict(path, file):

    with open(path+file,'rb') as handle:
        data = pk.loads(handle.read())

    return data
    
def read_cst_1d(path, file):
    '''
    Read CST plot data saved in ASCII .txt format
    '''
    vary=file.split('.')[0]
    X = []
    Y = []

    i=0
    with open(path+file) as f:
        for line in f:
            i+=1
            columns = line.split()
            if i==1: 
                varx = columns[0][0].lower()
            if i>2 and len(columns)>1:
                X.append(float(columns[0]))
                Y.append(float(columns[1]))

    X=np.array(X)
    Y=np.array(Y)

    return {varx: X , vary: Y}

def read_cst_3d(path):
    '''
    Pre-process the CST 3D field monitor output 

    Parameters
    -----------
    path: str 
        specify the path where the 3d Ez data folder is.

    Outputs
    --------
    cst.inp: dict
        pickle dict containing the domain and time info
    Ez.h5: .h5
        file containing the Ez(x,y,z) matrix for every timestep
    '''

    path = path + '3d/'
    hf_name = 'Ez.h5'

    # Rename files with E-02, E-03
    for file in glob.glob(path +'*E-02.txt'): 
        file=file.split(path)
        title=file[1].split('_')
        num=title[1].split('E')
        num[0]=float(num[0])/100

        ntitle=title[0]+'_'+str(num[0])+'.txt'
        os.rename(path+file[1], path+ntitle)

    for file in glob.glob(path +'*E-03.txt'): 
        file=file.split(path)
        title=file[1].split('_')
        num=title[1].split('E')
        num[0]=float(num[0])/1000

        ntitle=title[0]+'_'+str(num[0])+'.txt'
        os.rename(path+file[1], path+ntitle)

    for file in glob.glob(path +'*_0.txt'): 
        file=file.split(path)
        title=file[1].split('_')
        num=title[1].split('.')
        num[0]=float(num[0])

        ntitle=title[0]+'_'+str(num[0])+'.txt'
        os.rename(path+file[1], path+ntitle)

    def sorter(item):
        num=item.split(path)[1].split('_')[1].split('.txt')[0]
        return float(num)
    fnames = sorted(glob.glob(path+'*.txt'), key=sorter)
    #fnames = sorted(glob.glob(path+'*.txt'))

    #Get the number of longitudinal and transverse cells used for Ez
    i=0
    with open(fnames[0]) as f:
        lines=f.readlines()
        n_rows = len(lines)-3 #n of rows minus the header
        x1=lines[3].split()[0]

        while True:
            i+=1
            x2=lines[i+3].split()[0]
            if x1==x2:
                break

    n_transverse_cells=i
    n_longitudinal_cells=int(n_rows/(n_transverse_cells**2))

    # Create h5 file 
    if os.path.exists(path+hf_name):
        os.remove(path+hf_name)

    hf_Ez = h5py.File(path+hf_name, 'w')

    # Initialize variables
    Ez=np.zeros((n_transverse_cells, n_transverse_cells, n_longitudinal_cells))
    x=np.zeros((n_transverse_cells))
    y=np.zeros((n_transverse_cells))
    z=np.zeros((n_longitudinal_cells))
    t=[]

    nsteps, i, j, k = 0, 0, 0, 0
    skip=-4 #number of rows to skip
    rows=skip 

    # Start scan
    for file in fnames:
        print('[PROGRESS] Scanning file '+ file + '...')
        title=file.split(path)
        title2=title[1].split('_')
        num=title2[1].split('.txt')
        t.append(float(num[0])*1e-9)

        with open(file) as f:
            for line in f:
                rows+=1
                columns = line.split()

                if rows>=0 and len(columns)>1:
                    k=int(rows/n_transverse_cells**2)
                    j=int(rows/n_transverse_cells-n_transverse_cells*k)
                    i=int(rows-j*n_transverse_cells-k*n_transverse_cells**2) 
                    
                    Ez[i,j,k]=float(columns[5])
                    x[i]=float(columns[0])
                    y[j]=float(columns[1])
                    z[k]=float(columns[2])

        if nsteps == 0:
            prefix='0'*5
            hf_Ez.create_dataset('Ez_'+prefix+str(nsteps), data=Ez)
        else:
            prefix='0'*(5-int(np.log10(nsteps)))
            hf_Ez.create_dataset('Ez_'+prefix+str(nsteps), data=Ez)

        i, j, k = 0, 0, 0          
        rows=skip
        nsteps+=1

        #close file
        f.close()

    hf_Ez.close()    

    print('[PROGRESS] Finished scanning files')
    print('[INFO] Ez field is stored in a matrix with shape '+str(Ez.shape)+' in '+str(int(nsteps))+' datasets')
    print('[! OUT] hdf5 file'+hf_name+'succesfully generated')

    data = {}

    #Save x,y,z,t in dictionary
    data['x'] = x
    data['y'] = y 
    data['z'] = z
    data['t'] = np.array(t)

    with open(path+'cst.inp', 'wb') as fp:
        pk.dump(data, fp)

    #hf, dataset = read_Ez(path)
    #return data, hf, dataset


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
        n_step.append(int(key.split('_')[1]))

    # get size of matrix
    Ez_0=hf.get(dataset[0])
    shapex=Ez_0.shape[0]  
    shapey=Ez_0.shape[1] 
    shapez=Ez_0.shape[2] 

    print('[INFO] Ez field is stored in a matrix with shape '+str(Ez_0.shape)+' in '+str(int(size_hf))+' datasets')

    return hf, dataset

def read_pbci(path, file):

    case=file.split('.')[1]
    data = {}

    if case == 'potential':
        s, lambdas, WPx, WPy, WPz = [], [], [], [], []
        i=0
        with open(path+file) as file:
            for line in file:
                i+=1
                columns = line.split()

                if i>1 and len(columns)>1:

                    s.append(float(columns[0]))
                    lambdas.append(float(columns[1]))
                    WPx.append(float(columns[2]))
                    WPy.append(float(columns[3]))
                    WPz.append(float(columns[4]))

        data['s'] = np.array(s) #[xm]
        data['lambda'] = np.array(lambdas) #[1/xm]
        data['WPx'] = np.array(WPx)  #[V/pC]
        data['WPy'] = np.array(WPy)  #[V/pC]
        data['WPz'] = np.array(WPz)  #[V/pC]

    if case == 'impedance':
        f, lambdaf, Zx, Zy, Zz = [], [], [], [], []
        i=0
        with open(path+file) as file:
            for line in file:
                i+=1
                columns = line.split()

                if i>1 and len(columns)>1:

                    f.append(float(columns[0]))
                    lambdaf.append(float(columns[1]))
                    Zx.append(float(columns[2])+1j*float(columns[3]))
                    Zy.append(float(columns[4])+1j*float(columns[5]))
                    Zz.append(float(columns[6])+1j*float(columns[7]))

        data['f'] = np.array(f)  #[XHz]
        data['lambdaf'] = np.array(lambdaf)
        data['Zx'] = np.array(Zx)   #[Ohm]
        data['Zy'] = np.array(Zy)   #[Ohm]
        data['Zz'] = np.array(Zz)   #[Ohm]

    return data

def plot_Ez(path, t, point=(0,0,0), z=None, x=None, y=None):
    
    # get Ez
    hf, dataset = read_Ez(path)
    Ez_t = []

    # parse input
    xx = point[0]
    yy = point[1]
    zz = point[2]

    # plot default (0,0,0,t)
    if z == None and x == None and y == None:
        for n in range(len(dataset)):
            Ez=hf.get(dataset[n]) # [V/m]
            Ez_t.append(np.array(Ez[Ez.shape[0]//2, Ez.shape[1]//2,Ez.shape[2]//2]))

    # plot at given point
    else:
        # get index
        ix = min(range(len(x)), key=lambda i: abs(x[i]-xx))
        iy = min(range(len(y)), key=lambda i: abs(y[i]-yy))
        iz = min(range(len(z)), key=lambda i: abs(z[i]-zz))

        for n in range(len(dataset)):
            Ez=hf.get(dataset[n]) # [V/m]
            Ez_t.append(Ez[ix, iy, iz])

    Ez_t = np.array(Ez_t)
    label = 'Ez('+str(xx)+','+str(yy)+','+str(zz)+', t)'

    # plot 
    fig = plt.figure(1, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()
    ax.plot(t*1e9, Ez_t, lw=1.2, c='g', label = label)
    ax.set(title='Electric field '+label,
        xlabel='t [ns]',
        ylabel='Ez [V/m]',
        )

    ax.legend(loc='upper right')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()
    fig.savefig(path+label+'.png', bbox_inches='tight')


def animate_Ez(path, filename='Ez.h5', flag_charge_dist=True, flag_transverse_field=False):
    '''
    Creates an animated plot showing the Ez field along the z axis for every timestep

    Parameters:
    -----------
    -flag_charge_dist=True [def]: plots the passing beam charge distribution 
    -flag_compare_cst=True : add the comparison with CST field in cst dict
    -flag_transverse_field=True : add the Ez field in adjacent transverse cells Ez1(0+dx, 0+dy, z), Ez2(0+2dx, 0+2dy, z)
    '''

    # Read data
    hf, dataset = read_Ez(path, filename)

    data =  read_dict(path, 'warpx.in')

    t = data.get('t')               #simulated time [s]
    z = data.get('z')               #z axis values  [m]
    charge_dist = data.get('charge_dist')
    z0 = data.get('z0')             #full domain length (+pmls) [m]
    unit = data.get('unit')

    # Extract field on axis Ez (z,t)
    Ez0=[]
    for n in range(len(dataset)):
        Ez=hf.get(dataset[n]) # [V/m]
        Ez0.append(np.array(Ez[Ez.shape[0]//2, Ez.shape[1]//2,:])) # [V/m]

    Ez0=np.transpose(np.array(Ez0))

    if flag_transverse_field:
        Ez1=[]
        Ez2=[]
        for n in range(len(dataset)):
            Ez=hf.get(dataset[n]) # [V/m]
            Ez1.append(np.array(Ez[Ez.shape[0]//2+1, Ez.shape[1]//2+1,:])) # 1st adjacent cell Ez [V/m]
            Ez2.append(np.array(Ez[Ez.shape[0]//2+2, Ez.shape[1]//2+2,:])) # 2nd adjacent cell Ez [V/m]

        Ez1=np.transpose(np.array(Ez1))
        Ez2=np.transpose(np.array(Ez2))

    plt.ion()
    n=0
    for n in range(10,1000):
        if n % 1 == 0:
            #--- Plot Ez along z axis 
            fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
            ax=fig.gca()
            if flag_charge_dist:
                ax.plot(np.array(z0)/unit, charge_dist[:,n]/np.max(charge_dist)*np.max(Ez0)*0.4, lw=1.3, color='r', label='$\lambda $') 

            if flag_transverse_field:
                ax.plot(z/unit, Ez1[:, n], color='seagreen', label='Ez(0+dx, 0+dy, z) WarpX')
                ax.plot(z/unit, Ez2[:, n], color='limegreen', label='Ez(0+2dx, 0+2dy, z) WarpX')

            ax.plot(z/unit, Ez0[:, n], color='g', label='Ez(0,0,z) WarpX')
            ax.set(title='Electric field at time = '+str(round(t[n]*1e9,2))+' ns | timestep '+str(n),
                    xlabel='z [mm]',
                    ylabel='E [V/m]',         
                    ylim=(-np.max(Ez0)*1.1,np.max(Ez0)*1.1),
                    xlim=(min(z)/unit,max(z)/unit),
                            )
            ax.legend(loc='best')
            ax.grid(True, color='gray', linewidth=0.2)
            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.clf()
    plt.close()

def plot_pbci(path):

    d1 = read_pbci(path, 'path1.potential')
    d2 = read_pbci(path, 'path1.impedance')
    data = {**d1, **d2}

    # WPz
    fig = plt.figure(1, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()
    ax.plot(data['s']/1e-3, data['WPz'], lw=1.2, c='orange', label = 'WPz(s) PBCI')
    ax.plot(data['s']/1e-3, data['Lambda']*max(data['WPz'])/max(data['Lambda']), c='r', label= '$\lambda$(s)')

    ax.set(title='Longitudinal Wake potential W||(s)',
            xlabel='s [mm]',
            ylabel='Wz(s) [V/pC]',
            )
    ax.legend(loc='upper right')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()
    fig.savefig(path+'WPz.png', bbox_inches='tight')

    # Zz
    fig = plt.figure(2, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()
    ax.plot(data['f']*1e-9, abs(data['Zz']), lw=1.2, c='b', label = 'abs Zz(w) PBCI')
    ax.plot(data['f']*1e-9, np.real(data['Zz']), lw=1.2, c='r', ls='--', label = 'Re Zz(w) PBCI')
    ax.plot(data['f']*1e-9, np.imag(data['Zz']), lw=1.2, c='g', ls='--', label = 'Im Zz(w) PBCI')

    ax.set( title='Longitudinal impedance Z||(w)',
            xlabel='f [GHz]',
            ylabel='Z||(w) [$\Omega$]',   
             )
    ax.legend(loc='upper left')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()
    fig.savefig(path+'Zz.png', bbox_inches='tight')

    # WPx, WPy
    fig = plt.figure(3, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()
    ax.plot(data['s']/1e-3, data['WPx'], lw=1.2, color='g', label='Wx⊥(s) PBCI')
    ax.plot(data['s']/1e-3, data['WPy'], lw=1.2, color='magenta', label='Wy⊥(s) PBCI')
    ax.set(title='Transverse Wake potential W⊥(s)',
            xlabel='s [mm]',
            ylabel='$W_{⊥}$ [V/pC]',
             )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()
    fig.savefig(path+'WPxy.png', bbox_inches='tight')

    #Zx, Zy 
    fig = plt.figure(3, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()

    ax.plot(data['f']*1e-9, abs(data['Zx']), lw=1.2, c='g', label = 'abs Zx(w) PBCI')
    ax.plot(data['f']*1e-9, np.real(data['Zx']), lw=1, c='g', ls=':', label = 'Re Zx(w) PBCI')
    ax.plot(data['f']*1e-9, np.imag(data['Zx']), lw=1, c='g', ls='--', label = 'Im Zx(w) PBCI')

    ax.plot(data['f']*1e-9, abs(data['Zy']), lw=1.2, c='magenta', label = 'abs Zx(w) PBCI')
    ax.plot(data['f']*1e-9, np.real(data['Zy']), lw=1, c='magenta', ls=':', label = 'Re Zy(w) PBCI')
    ax.plot(data['f']*1e-9, np.imag(data['Zy']), lw=1, c='magenta', ls='--', label = 'Im Zy(w) PBCI')

    ax.set(title='Transverse impedance Z⊥(w)',
            xlabel='f [GHz]',
            ylabel='Z⊥(w) [$\Omega$]',   
             )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()
    fig.savefig(path+'Zxy.png', bbox_inches='tight')


def triang_implicit(fun, BC=None, bbox=(-3,3)):
    ''' 
    Triangulates the surface of an implicit function f(x,y,z) = 0
    
    Parameters:
    -----------
    - fun: implicit function (plot where fn==0)
    - BC: class containing the implicit function
    - bbox: the x,y,and z limits of plotting box

    Returns:
    --------
    - F: matrix with shape(32,32,32) that only has values 
         where the implicit function is close to 0 and 
         otherwise in float('nan') 

    Requirements:
    ------------
    scikit-image
    ''' 
    from skimage import measure #

    # Define box limits
    bmin, bmax = bbox

    # Define the mesh 
    xl = np.linspace(bmin, bmax, 32)
    X, Y, Z = np.meshgrid(xl, xl, xl)

    #Define the field
    if BC == None:
        F = fun(X, Z, Y)

    else:
        F = fun(X, Z, Y, BC)

    # Obtain vertex and faces with the marching cubes algorithm
    verts, faces, normals, values = measure.marching_cubes(F, 0, spacing=[np.diff(xl)[0]]*3 )
    verts -= bmax

    # Plot using trisurf
    fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color='b')

    # Set plot properties
    ax.set_zlim3d(bmin,bmax)
    ax.set_xlim3d(bmin,bmax)
    ax.set_ylim3d(bmin,bmax)
    ax.grid(True, color='gray', linestyle=':')
    ax.set(title='Geometry defined by the implicit function', 
            xlabel='z [m]',
            ylabel='x [m]',
            zlabel='y [m]'
            )
    plt.show()

    return F

def implicit_function(x, y, z, BC): 
    ''' 
    Evaluates the expression parsed to the WarpX picmi.EmbeddedBoundary class.
    - x, y, z: variables to be defined by the plot function 
    - BC: custom class containing the implicit function and 
         parameters in a string + user defined kw arguments
    '''
    # read the class
    implicit_class=BC.implicit_function.split(';')
    params=BC.user_defined_kw

    args=implicit_class[0:-1]      #assignements
    implicitf=implicit_class[-1]   #implicit function
    vals=np.zeros((len(args), len(x), len(y), len(z)))

    # introduce kw parameters [NOT wOrKING]
    i=0
    for i in range(len(args)):
        for key in params.keys():  
            if key in args[i]:
                args[i]=args[i].replace(key, 'params[\''+key+'\']')
                

    # get the arguments value
    for i in range(len(args)):
        arg=args[i].split('=')
        vals[i, :, :, :]=eval(arg[1])
        implicitf=implicitf.replace(arg[0].strip(), 'vals['+str(i)+', :, :, :]')

    implicitf=implicitf.replace('max', 'np.maximum')
    
    return eval(implicitf) 

#--------------------------------------------------------------

if __name__ == "__main__":

    path = '/mnt/c/Users/elefu/Documents/CERN/PBCI_CERN/examples/collimator/'

    plot_pbci(path)
