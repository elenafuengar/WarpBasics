'''
Bucket class to manage Mixin classes

@date: Created on 20.10.2022
@author: Elena de la Fuente
'''

from wakis.reader import Reader
from wakis.plotting import Plotter
from wakis.solver import Solver

class Bucket(Reader, Plotter, Solver):
    '''Bucket class to encapsulate all Mixin classes
    '''
    def check_inputs(self): 
        '''Function for error handling [TODO]
        '''
        pass
  