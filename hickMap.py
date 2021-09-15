# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 23:20:12 2021

@author: alier
"""

import numpy as np
from rppp import PPP
from rppp import mtPPP
from MCMCadams import MCMCadams
from gfunc_est import pairs, gfuncest

import matplotlib.pyplot as plt

hickory = np.loadtxt("hickory.csv", delimiter=",")
maple = np.loadtxt("maple.csv", delimiter=",")



### beautiful plot

nI = hickory.shape[0]
nP = maple.shape[0]

lansat = np.concatenate((np.concatenate((hickory,maple)),
np.concatenate((np.full((nI,1),2),np.full((nP,1),3)))),axis=1)
# marks = np.concatenate((np.full(n2,r'$\clubsuit$'),np.full(n3,r'$\spadesuit$')))

coli = np.concatenate((np.full(nI,"tab:green"),np.full(nP,"tab:red")))


N = nI+nP

ind = np.arange(N)
np.random.shuffle(ind)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')


plt.xlim(0,1)
plt.ylim(0,1)


plt.scatter(hickory[:,0],hickory[:,1],label="hickory",s=5,c="tab:green", marker=r'$\clubsuit$')

plt.scatter(maple[:,0],maple[:,1],label="maple",s=5,c="tab:red", marker=r'$\clubsuit$')

plt.legend(bbox_to_anchor=(1, 0.8), markerscale=2)

plt.scatter(lansat[ind,0],lansat[ind,1],c=coli[ind],s=20, marker=r'$\clubsuit$')

# plt.show()
fig.savefig("hickMap.pdf", bbox_inches='tight')   

###


thinp = 0.1

hickoryPP = PPP(hickory)
hickoryPP.thin(thinp)

maplePP = PPP(maple)
maplePP.thin(thinp)





mtpp = mtPPP(np.array([hickoryPP,maplePP]))

mtpp.plot()



### mcmc


K = mtpp.K

lam_est = mtpp.nObs*(K+1)/K


size=30000
nInsDelMov = lam_est//10

var = 4
rho=5



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
mu=lam_est
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
N=1000

steps = np.linspace(0.0, 0.8, num=20)


tail = 10000
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


# N = 10000

# # scl = var
# # V = np.array([[1,-0.9],[-0.9,1]])*scl
# # V = np.array([[1,0.9],[0.9,1]])*scl
# # V = np.array([[1,0],[0,1]])*scl

# # mn = -2
# mu = mean_mu[0]

# # rho = 5


# # steps = np.arange(0,1.1,0.1)
# gs = gfuncest(N,V,mu,rho,steps)


fig, axs = plt.subplots(2, 2)
# ax.set_box_aspect(1)

axs[0, 0].plot(steps,gs_mean[:,0],  c="tab:green",  linestyle="dotted")
# plt.plot(steps,gs_lower[:,0], linestyle="dashed", c="tab:blue")
# plt.plot(steps,gs_higher[:,0], linestyle="dashed", c="tab:blue")
axs[0, 0].fill_between(steps, gs_lower[:,0], gs_higher[:,0], color="tab:green", alpha=0.3, linewidth=0)
# plt.show()

axs[0, 0].plot(steps,gs_mean[:,2],  c="tab:red",  linestyle="dotted")
# plt.plot(steps,gs_lower[:,2], linestyle="dashed", c="tab:orange")
# plt.plot(steps,gs_higher[:,2], linestyle="dashed", c="tab:orange")
axs[0, 0].fill_between(steps, gs_lower[:,2], gs_higher[:,2], color="tab:red", alpha=0.3, linewidth=0)
# plt.show()

axs[0, 0].plot(steps,gs_mean[:,1],  c="grey",  linestyle="dotted")
# plt.plot(steps,gs_lower[:,1], linestyle="dashed", c="black")
# plt.plot(steps,gs_higher[:,1], linestyle="dashed", c="black")
axs[0, 0].fill_between(steps, gs_lower[:,1], gs_higher[:,1], color="grey", alpha=0.3, linewidth=0)


axs[0, 1].plot(steps,gs_mean[:,2],  c="tab:red",  linestyle="dotted")
# plt.plot(steps,gs_lower[:,2], linestyle="dashed", c="tab:orange")
# plt.plot(steps,gs_higher[:,2], linestyle="dashed", c="tab:orange")
axs[0, 1].fill_between(steps, gs_lower[:,2], gs_higher[:,2], color="tab:red", alpha=0.3, linewidth=0)
# plt.show()

axs[1, 0].plot(steps,gs_mean[:,0],  c="tab:green",  linestyle="dotted")
# plt.plot(steps,gs_lower[:,0], linestyle="dashed", c="tab:blue")
# plt.plot(steps,gs_higher[:,0], linestyle="dashed", c="tab:blue")
axs[1, 0].fill_between(steps, gs_lower[:,0], gs_higher[:,0], color="tab:green", alpha=0.3, linewidth=0)
# plt.show()


axs[1, 1].plot(steps,gs_mean[:,1],  c="grey",  linestyle="dotted")
# plt.plot(steps,gs_lower[:,1], linestyle="dashed", c="black")
# plt.plot(steps,gs_higher[:,1], linestyle="dashed", c="black")
axs[1, 1].fill_between(steps, gs_lower[:,1], gs_higher[:,1], color="grey", alpha=0.3, linewidth=0)


# fig.savefig("hickMapInt.pdf", bbox_inches='tight')  
plt.show()



# plt.plot(steps,gs[:,0], c="tab:blue")
# # plt.show()
# plt.plot(steps,gs[:,2], c="tab:orange")
# # plt.show()
# plt.plot(steps,gs[:,1], c="grey")
# # plt.plot(steps,gs[:,1], linestyle="dashed", c="tab:blue")



















