# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:31:35 2021

@author: alier
"""

from MCMCadams import MCMCadams
import numpy as np
from rppp import PPP
from rGP import GP
from rGP import gaussianCov 
from rGP import zeroMean



def fct(x):
    return(np.exp((-x[:,0]**2-x[:,1]**2)/0.3))
# def fct(x):
#     return(np.exp(-(0.25-np.sqrt((x[:,0]-0.5)**2+(x[:,1]-0.5)**2))**2/0.003)*0.8+0.10)
# def fct(x):
#     return(np.exp(-np.minimum((x[:,0]-0.5)**2,(x[:,1]-0.5)**2)/0.007)*0.8+0.10)

lam_sim=800

pointpo = PPP.randomNonHomog(lam_sim,fct)
pointpo.plot()


newGP = GP(zeroMean,gaussianCov(2,0.5))

niter=400

import time

t0 = time.time()
thinLoc,thinVal,obsVal,lams = MCMCadams(niter,lam_sim,newGP,pointpo,100,10,0.1,10,lam_sim,10000)
t1 = time.time()

total1 = t1-t0