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
rho=5

locs = PPP.randomHomog(lam).loc


newGP = GP(zeroMean,expCov(1,rho))

U = newGP.covMatrix(locs)

var=5

# V = np.array([[1]])*var
V = np.array([[1,-0.99],[-0.99,1]])*var
# V = np.array([[1,0.99],[0.99,1]])*var
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
locs2 = locs[colours == 1]





### make an mtPP format object


pp1 = PPP(locs1)
pp2 = PPP(locs2)


pps = np.array([pp1,pp2])

mtpp = mtPPP(pps)

mtpp.plot()



### initialize other MCMC parameters



K = mtpp.K

lam_est = mtpp.nObs*(K+1)/K


size=1000
nInsDelMov = lam_est//10





n=10/var

# V_mc=np.linalg.inv(V/var)/10
V_mc=np.identity(K)/10

# T_init = np.linalg.inv(V)
T_init=np.identity(K)/var


kappa=10
delta=0.1
L=10
mu=lam
sigma2=1000
a=rho*10
b=10

p=False


import time

t0 = time.time()
lams,rhos,Ts, Nthins = MCMCadams(size,lam_est,a/b,T_init,mtpp,nInsDelMov,kappa,delta,L,
                                 mu,sigma2,p,a,b,n,V_mc,diagnostics=True,res=25,thin=10,GP_mom_scale=4,range_mom_scale=7)
t1 = time.time()

tt1 = t1-t0




### trace plots

plt.plot(lams)
plt.show()

plt.plot(rhos)
plt.show()


### correlations

Covs = np.array([np.linalg.inv(t) for t in Ts])

plt.plot(Covs[:,0,0])
plt.show()

plt.plot(Ts[:,0,0])
plt.show()

plt.plot(Nthins)
plt.show()


corr01 = Covs[:,0,1]/np.sqrt(Covs[:,0,0]*Covs[:,1,1])
plt.plot(corr01)
plt.show()


# plt.plot(Ts[:,0,0])
# plt.show()

### GP values trace

# nObs = mtpp.nObs


# obsGP = np.empty(shape=(size-1,nObs,K))

# i=0
# while(i < size-1):
#     values = np.loadtxt("values"+str(i)+".csv", delimiter=",")
#     # np.savetxt("resGP"+str(i)+".csv",lams[i+1]*expit(newGP.rCondGP(gridLoc,locations,np.transpose([values]))) ,delimiter=",")

#     obsGP[i] = np.transpose([values[0:nObs]])

    
#     print(i)
#     i+=1
    
    
# fig, axs = plt.subplots(nObs,figsize=(10,nObs))    
    
# i=0
# while(i < nObs):
#     axs[i].plot(obsGP[:,i])

    
#     print(i)
#     i+=1    

# # plt.show()
# fig.savefig("0GPtraces.pdf", bbox_inches='tight')

