# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:54:01 2021

@author: alier
"""

from rppp import PPP
import numpy as np

### usage example

lam=1000

pointpo = PPP.randomHomog(lam)
pointpo.plot()

def fct(x):
    return(np.exp(-np.minimum((x[:,0]-0.5)**2,(x[:,1]-0.5)**2)/0.003))


pointpo = PPP.randomNonHomog(lam,fct)
pointpo.plot()


def fct(x):
    return(np.exp(-(0.25-np.sqrt((x[:,0]-0.5)**2+(x[:,1]-0.5)**2))**2/0.003))

pointpo = PPP.randomNonHomog(lam,fct)
pointpo.plot()