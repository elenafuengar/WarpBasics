'''
Reader module containing methods 
to pre-process and parse different 
input file formats from 
the supported EM solvers

@date: Created on 01.11.2022
@author: Elena de la Fuente
'''

import os
import glob
import shutil
import json as js
import pickle as pk
from tqdm import tqdm

#dependencies 
import numpy as np
import h5py 

class Reader():
    '''Mixin class to encapsulate input reading methods
    '''

    def read_warpx(self, filename=None):

        if filename is None:
            filename = self.input_file

        exts = ['pk', 'pickle', 'out', 'dat']
        ext = filename.split('.')[-1]

        if ext == 'js' or ext == 'json':
            with open(filename, 'r') as f:
                d = {k: np.array(v) for k, v in js.loads(f.read()).items()}
        elif ext in exts:
            with open(filename, 'rb') as f:
                d = pk.load(f)

        for key, val in d.items():
            setattr(self, key, val)


    def read_cst_1d(self, file, path=None):
        '''
        Read CST plot data saved in ASCII .txt format

        Parameters:
        ---
        file : str
            Name of the .txt file to read. Example: 'lambda.txt' 
        path : :obj: `str`, optional
            Absolute path to file. Deafult is cwd
        '''
        if path is None:
            path=self.path

        X = []
        Y = []

        i=0
        with open(path+file) as f:
            for line in f:
                i+=1
                columns = line.split()

                if i>1 and len(columns)>1:

                    X.append(float(columns[0]))
                    Y.append(float(columns[1]))

        X=np.array(X)
        Y=np.array(Y)
        return {'X':X , 'Y': Y}


    def read_Ez(self, filename=None, return_=False):
        '''
        Read the Ez.h5 file containing the Ez field information
        '''

        if filename is None:
            filename = self.Ez_file

        hf = h5py.File(filename, 'r')
        self.log.info(f'Reading h5 file {filename}' )
        self.log.debug('Size of the file: ' + str(round((os.path.getsize(filename)/10**9),2))+' Gb')

        #Set attributes
        self.Ez_hf = hf
        self.Ez_file = filename
        if 'x' in hf.keys():
            self.xf = np.array(hf['x'])
        if 'y' in hf.keys():
            self.yf = np.array(hf['y'])
        if 'z' in hf.keys():
            self.zf = np.array(hf['z'])
        if 't' in hf.keys():
            self.t = np.array(hf['t'])

        if return_:
            return hf

    def read_cst_3d(self, path=None, folder='3d', filename='Ez.h5', units=1e-3, save_pickle=False, save_json=False, **kwargs):
        '''
        Read CST 3d exports folder and store the
        Ez field information into a matrix Ez(x,y,z) 
        for every timestep into a single `.h5` file

        Parameters
        ----------
        path: str, default None
            Path to the field data 
        folder: str, default '3d'
            Folder containing the CST field data .txt files
        filename: str, default 'Ez.h5'
            Name of the h5 file that will be generated
        save: bool, default True
            Flag to save the field and domain data with pickle
        save_json: bool, default False
            Flag to save the field and domain data in json format

        '''  
        if path is None:
            path = self.path + folder + '/'



        # Rename files with E-02, E-03
        for file in glob.glob(path +'*E-02.txt'): 
            file=file.split(path)
            title=file[1].split('_')
            num=title[1].split('E')
            num[0]=float(num[0])/100

            ntitle=title[0]+'_'+str(num[0])+'.txt'
            shutil.copy(path+file[1], path+file[1]+'.old')
            os.rename(path+file[1], path+ntitle)

        for file in glob.glob(path +'*E-03.txt'): 
            file=file.split(path)
            title=file[1].split('_')
            num=title[1].split('E')
            num[0]=float(num[0])/1000

            ntitle=title[0]+'_'+str(num[0])+'.txt'
            shutil.copy(path+file[1], path+file[1]+'.old')
            os.rename(path+file[1], path+ntitle)

        for file in glob.glob(path +'*_0.txt'): 
            file=file.split(path)
            title=file[1].split('_')
            num=title[1].split('.')
            num[0]=float(num[0])

            ntitle=title[0]+'_'+str(num[0])+'.txt'
            shutil.copy(path+file[1], path+file[1]+'.old')
            os.rename(path+file[1], path+ntitle)

        #sort
        def sorter(item):
            num=item.split(path)[1].split('_')[1].split('.txt')[0]
            return float(num)
        fnames = sorted(glob.glob(path+'*.txt'), key=sorter)

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
        if os.path.exists(path+filename):
            os.remove(path+filename)

        hf = h5py.File(path+filename, 'w')

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
        self.log.info(f'Scanning files in {path}:')
        for file in tqdm(fnames):
            #self.log.debug('Scanning file '+ file + '...')
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

                        if k>= n_longitudinal_cells:
                            k = int(n_longitudinal_cells-1)

                        Ez[i,j,k]=float(columns[5])

                        x[i]=float(columns[0])*units
                        y[j]=float(columns[1])*units
                        z[k]=float(columns[2])*units

            if nsteps == 0:
                prefix='0'*5
                hf.create_dataset('Ez_'+prefix+str(nsteps), data=Ez)
            else:
                prefix='0'*(5-int(np.log10(nsteps)))
                hf.create_dataset('Ez_'+prefix+str(nsteps), data=Ez)

            i, j, k = 0, 0, 0          
            rows=skip
            nsteps+=1

            #close file
            f.close()

        hf['x'] = x
        hf['y'] = y
        hf['z'] = z
        hf['t'] = t

        hf.close()

        #set field info
        self.log.debug('Ez field is stored in a matrix with shape '+str(Ez.shape)+' in '+str(int(nsteps))+' datasets')
        self.log.info(f'Finished scanning files - hdf5 file {filename} succesfully generated')

        #Update self
        self.xf = x
        self.yf = y 
        self.zf = z
        self.t = np.array(t)

        for key, val in kwargs.items():
            setattr(self, key, val)

        if save_pickle:
            ext = 'dat'
            d = {'xf': x, 'yf': y, 'zf': z, 't': t}
            with open('cst.' + ext, 'wb') as f:
                pk.dump(d, f)
            self.log.info('"cst.' + ext +'" file succesfully generated') 

        if save_json:
            ext = 'json'
            d = {'xf': x, 'yf': y, 'zf': z, 't': t}
            j = json.dumps({k: d[k].tolist() for k in keys})
            with open('cst.' + ext, 'w') as f:
                json.dump(j, f)
            self.log.info('"cst.'+ ext +'" file succesfully generated') 



