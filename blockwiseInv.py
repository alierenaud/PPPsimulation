# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:10:04 2021

@author: alier
"""

import numpy as np
import numpy.linalg
import scipy as sp
from rGP import GP
from rGP import gaussianCov
from rppp import PPP
from numpy import random


def zeroMean(x):
    return(0) 


thisGP = GP(zeroMean,gaussianCov(2,0.5))
lam=2000

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

    A = np.array([[Sigma[0,0]]])
    B = np.array([Sigma[0,1:Sigma.shape[0]]])
    C = np.transpose(B)


    BD = B@Sigma22_inv
    BDC = BD@C
    DC = np.transpose(BD)
    AmBDC = 1/(A-BDC) 
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

#### deletion



thisGP = GP(zeroMean,gaussianCov(2,0.5))
lam=2000

pointpo = PPP.randomHomog(lam)
pointpo.plot()

import time

t0 = time.time()
Sigma = thisGP.covMatrix(pointpo.loc)
t1 = time.time()

total0 = t1-t0
total0


Sigma_inv = np.linalg.inv(Sigma)


### delete ith row and cloumn


n = Sigma.shape[0]
    
i = random.choice(np.array(range(0,n)))


Sigma_del = np.delete(np.delete(Sigma, i, 0), i, 1)


def woodDelInv(Sigma,Sigma_inv,i):
    
    n = Sigma.shape[0]

    V = np.concatenate(([Sigma[i,:]],[np.zeros(n)]))
    V[:,i] = [0,1] 

    U = np.transpose(V)[:,::-1]
    

    B_inv = Sigma_inv + Sigma_inv@U@np.linalg.inv(np.identity(2)-V@Sigma_inv@U)@V@Sigma_inv

    B_inv_del = np.delete(np.delete(B_inv,i,0),i,1)


    
    return(B_inv_del)


t0 = time.time()
Sigma_del_inv = np.linalg.inv(Sigma_del)
t1 = time.time()

total1 = t1-t0


t0 = time.time()
Sigma_del_inv_wood = woodDelInv(Sigma,Sigma_inv,i)
t1 = time.time()

total2 = t1-t0


#### cholesky vs inv vs solve

thisGP = GP(zeroMean,gaussianCov(2,0.5))
lam=2000

pointpo = PPP.randomHomog(lam)
pointpo.plot()

n = pointpo.loc.shape[0]

import time

t0 = time.time()
Sigma = thisGP.covMatrix(pointpo.loc)
t1 = time.time()

total0 = t1-t0
total0


t0 = time.time()
Sigma_inv = np.linalg.inv(Sigma)
t1 = time.time()

total1 = t1-t0
total1


t0 = time.time()
A = np.linalg.cholesky(Sigma)
t1 = time.time()

total2 = t1-t0
total2


t0 = time.time()
A_inv = sp.linalg.solve_triangular(A,np.identity(n),lower=True)
t1 = time.time()

total3 = t1-t0
total3

A_inv@A








