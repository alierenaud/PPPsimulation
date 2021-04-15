# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:10:04 2021

@author: alier
"""

import numpy as np
# import numpy.linalg
from rGP import GP
from rGP import gaussianCov
from rppp import PPP

def zeroMean(x):
    return(0) 


thisGP = GP(zeroMean,gaussianCov(2,0.5))
lam=1000

pointpo = PPP.randomHomog(lam)
pointpo.plot()

import time

t0 = time.time()
Sigma = thisGP.covMatrix(pointpo.loc)
t1 = time.time()

total0 = t1-t0
total0


Sigma22 = Sigma[1:Sigma.shape[0],1:Sigma.shape[0]]
Sigma22_inv = np.linalg.inv(Sigma22)
Sigma22 @ Sigma22_inv


def blockwiseInv(Sigma,Sigma22_inv):

    a = Sigma[0,0]
    B = Sigma[0,1:Sigma.shape[0]]
    C = np.transpose(B)


    BD = B@Sigma22_inv
    BDC = BD@C
    DC = np.transpose(BD)
    AmBDC = 1/(a-BDC) 
    AmBDCBD = AmBDC@BD

    Sigma_inv = np.block([[AmBDC,-AmBDCBD],[-DC@AmBDC,Sigma22_inv + DC@AmBDCBD]])
    
    return(Sigma_inv)





t0 = time.time()
Sigma_inv_block = blockwiseInv(Sigma,Sigma22_inv)
t1 = time.time()

total1=t1-t0


Sigma@Sigma_inv_block


t0 = time.time()
Sigma_inv = np.linalg.inv(Sigma)
t1 = time.time()

total2=t1-t0


Sigma@Sigma_inv

Sigma_inv_block
Sigma_inv




