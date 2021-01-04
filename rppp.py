# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 12:42:08 2020

@author: alierenaud
"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

class PPP:  
    def __init__(self, loc):
        self.loc = loc

    @classmethod
    def randomHomog(cls,lam):
        N = int(random.poisson(lam=lam, size=1))
        loc = random.uniform(size=(N, 2))
        return cls(loc)
    
    @classmethod
    def randomNonHomog(cls,lam,fct):
        newPPP = cls.randomHomog(lam)
        index = np.greater(fct(newPPP.loc),random.uniform(size=newPPP.loc.shape[0]))
        newPPP.loc = newPPP.loc[index]
        return(newPPP)
        
    
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')

        plt.plot(self.loc[:,0],self.loc[:,1], 'o')
        plt.xlim(0,1)
        plt.ylim(0,1)

        plt.show()














