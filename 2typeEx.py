# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:53:01 2021

@author: alier
"""

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt


from rppp import PPP
from rppp import mtPPP
from rGP import GP
from rGP import expCov
from rGP import zeroMean

from MCMCadams import MCMCadams

from scipy.stats import matrix_normal



######
### Examples and testing of the multiType SGCD MCMC alogrithm
######

## Generate a mtype SGCD

lam=500
rho=10

locs = PPP.randomHomog(lam).loc


newGP = GP(zeroMean,expCov(1,rho))

U = newGP.covMatrix(locs)

var=1

V = np.array([[1]])*var
# V = np.array([[1,-0.9],[-0.9,1]])*var
# V = np.array([[1,0.9],[0.9,1]])*var
# V = np.array([[1,0],[0,1]])*var



X = matrix_normal.rvs(rowcov=U, colcov=V)

def multExpit(x):
    N = np.sum(np.exp(x))
    probs = np.array([np.exp(i)/(1+N) for i in x])
    return(np.append(probs,1-np.sum(probs)))
        
        
probs = np.array([multExpit(x) for x in X])

nch = probs.shape[1]

colours = np.array([np.random.choice(nch,p=p) for p in probs])

locs1 = locs[colours == 0]
# locs2 = locs[colours == 1]





### make an mtPP format object


pp1 = PPP(locs1)
# pp2 = PPP(locs2)


pps = np.array([pp1])

mtpp = mtPPP(pps)

mtpp.plot()



### initialize other MCMC parameters



K = mtpp.K

lam_est = mtpp.nObs*(K+1)/K


size=1000
nInsDelMov = lam_est//10





n=500

V_mc=V*(n-K-1)
# V_mc=np.identity(K)/var/n

T_init = V
# T_init=np.identity(K)/var


kappa=10
delta=0.1
L=10
mu=lam
sigma2=100
a=rho*100
b=100

p=False


import time

t0 = time.time()
locations,values,lams,rhos,Vs = MCMCadams(size,lam_est,a/b,T_init,mtpp,nInsDelMov,kappa,delta,L,mu,sigma2,p,a,b,n,V_mc)
t1 = time.time()

tt1 = t1-t0




### trace plots

plt.plot(lams)
plt.show()

plt.plot(rhos)
plt.show()


### correlations



plt.plot(Vs[:,0,0])
plt.show()

plt.plot(Vs[:,1,1])
plt.show()

plt.plot(Vs[:,0,1])
plt.show()

plt.plot(Vs[:,0,1]/np.sqrt(Vs[:,0,0]*Vs[:,1,1]))
plt.show()


