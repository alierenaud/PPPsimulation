# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:53:01 2021

@author: alier
"""

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from gfunc_est import gfuncest
from gfunc_est import pairs
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

var=4

# V = np.array([[1]])*var
V = np.array([[1,-0.9],[-0.9,1]])*var
# V = np.array([[1,0.9],[0.9,1]])*var
# V = np.array([[1,0.5],[0.5,1]])*var
# V = np.array([[1,-4.5],[-4.5,25]])*var

mu=-2
# mu=[0,-2]

X = matrix_normal.rvs(rowcov=U, colcov=V) + mu

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

# pps = np.array([pp1])
pps = np.array([pp1,pp2])

mtpp = mtPPP(pps)

mtpp.plot()



### initialize other MCMC parameters



K = mtpp.K

lam_est = mtpp.nObs*(K+1)/K


size=10000
nInsDelMov = lam_est//10





n=2

V_mc=np.linalg.inv(np.array([[1/100,0],[0,1/100]])) ### attempt at neutral
# V_mc=np.linalg.inv(V/var)/10
# V_mc=np.identity(K)/10

# T_init=np.linalg.inv(np.array([[1,0.5],[0.5,1]]))/var
# T_init = np.linalg.inv(V)
T_init=np.identity(K)/var


kappa=10
delta=0.05
L=10
mu=lam
sigma2=1000
a=rho*10
b=10
mu_init = np.array([-2,-2])
mean_mu = np.array([[-2,-2]])
var_mu = 4

p=False


import time

t0 = time.time()
lams,rhos,Ts,mus, Nthins = MCMCadams(size,lam_est,a/b,T_init,mtpp,nInsDelMov,kappa,delta,L,
                                 mu,sigma2,p,a,b,n,V_mc,mu_init,mean_mu,var_mu,diagnostics=False,res=25,thin=10,GP_mom_scale=5,range_mom_scale=5)
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
plt.plot(Covs[:,1,1])
plt.plot(Covs[:,0,1], c="grey")
plt.show()

plt.plot(Ts[:,0,0])
plt.plot(Ts[:,1,1])
plt.plot(Ts[:,0,1], c="grey")
plt.show()

plt.plot(mus[:,:,0])
# plt.show()
plt.plot(mus[:,:,1])
plt.show()

plt.plot(Nthins)
plt.show()


corr01 = Covs[:,0,1]/np.sqrt(Covs[:,0,0]*Covs[:,1,1])
plt.plot(corr01)
plt.show()

#### g function
N=100

steps = np.linspace(0.0, 1.0, num=50)


tail = 4000
pairs(K)
gs = np.empty(shape=(tail,steps.shape[0],pairs(K).shape[0]))
i=0
while(i<tail):
    
    gs[i] = gfuncest(N,Covs[size-tail+i],mus[size-tail+i,0],rhos[size-tail+i],steps)
    print(i)
    i+=1


gs_mean = np.mean(gs,axis=0)
gs_lower = np.quantile(gs,q=0.1,axis=0)
gs_higher = np.quantile(gs,q=0.9,axis=0)


N = 10000

# scl = var
# V = np.array([[1,-0.9],[-0.9,1]])*scl
# V = np.array([[1,0.9],[0.9,1]])*scl
# V = np.array([[1,0],[0,1]])*scl

# mn = -2
mu = mean_mu[0]

# rho = 5


# steps = np.arange(0,1.1,0.1)
gs = gfuncest(N,V,mu,rho,steps)




plt.plot(steps,gs_mean[:,0],  c="tab:blue",  linestyle="dotted")
# plt.plot(steps,gs_lower[:,0], linestyle="dashed", c="tab:blue")
# plt.plot(steps,gs_higher[:,0], linestyle="dashed", c="tab:blue")
plt.fill_between(steps, gs_lower[:,0], gs_higher[:,0], color="tab:blue", alpha=0.3, linewidth=0)
# plt.show()

plt.plot(steps,gs_mean[:,2],  c="tab:orange",  linestyle="dotted")
# plt.plot(steps,gs_lower[:,2], linestyle="dashed", c="tab:orange")
# plt.plot(steps,gs_higher[:,2], linestyle="dashed", c="tab:orange")
plt.fill_between(steps, gs_lower[:,2], gs_higher[:,2], color="tab:orange", alpha=0.3, linewidth=0)
# plt.show()

plt.plot(steps,gs_mean[:,1],  c="grey",  linestyle="dotted")
# plt.plot(steps,gs_lower[:,1], linestyle="dashed", c="black")
# plt.plot(steps,gs_higher[:,1], linestyle="dashed", c="black")
plt.fill_between(steps, gs_lower[:,1], gs_higher[:,1], color="grey", alpha=0.3, linewidth=0)




plt.plot(steps,gs[:,0], c="tab:blue")
# plt.show()
plt.plot(steps,gs[:,2], c="tab:orange")
# plt.show()
plt.plot(steps,gs[:,1], c="grey")
# plt.plot(steps,gs[:,1], linestyle="dashed", c="tab:blue")
plt.show()



# plt.show()



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

