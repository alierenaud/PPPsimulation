# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:03:36 2021

@author: alier
"""

import numpy as np
import matplotlib.pyplot as plt
# import scipy as sp


def multexpit(x):
    N = np.sum(np.exp(x))
    return np.exp(x)/(1+N)

def pairs(K):
    arr = np.empty(shape=(int((K+1)*K/2),2), dtype=int)
    i = 0
    c = 0
    while i<K:
        j=i
        while j<K:
            arr[c] = [i,j]
            c += 1
            j += 1
        i += 1
    return arr

def g0(N,V,mu):
    
    B = np.linalg.cholesky(V)

    K = V.shape[0]
    
    Y = np.random.normal(size=(N,K))@np.transpose(B) + mu
    
    Z = np.array([multexpit(y) for y in Y])
    
    
    means = np.mean(Z, axis=0)

    return np.array([np.mean(Z[:,pr[0]]*Z[:,pr[1]])/(means[pr[0]]*means[pr[1]]) for pr in pairs(K)])

# N = 1000

# V = np.array([[1,-0.9],[-0.9,1]])

# mu = np.array([-2,-2])


# g0(10000,V,mu)

def gd(N,V,mu,rho,d):
    
    B = np.linalg.cholesky(np.kron([[1,np.exp(-rho*d)],[np.exp(-rho*d),1]],V))

    K = V.shape[0]
    
    Y = np.random.normal(size=(N,2*K))@np.transpose(B) + np.concatenate((mu,mu))
    
    
    
    Zeta = np.array([multexpit(y) for y in Y[:,:K]])
    Zxi = np.array([multexpit(y) for y in Y[:,K:]])
    
    
    meansEta = np.mean(Zeta, axis=0)
    meansXi = np.mean(Zxi, axis=0)
    

    return np.array([np.mean(Zeta[:,pr[0]]*Zxi[:,pr[1]])/(meansEta[pr[0]]*meansXi[pr[1]]) for pr in pairs(K)])


# N = 1000

# V = np.array([[1,-0.9],[-0.9,1]])

# mu = np.array([-2,-2])

# rho = 5


# g0(N,V,mu)
# gd(N,V,mu,rho,0.2)
# gd(N,V,mu,rho,0.4)
# gd(N,V,mu,rho,0.6)
# gd(N,V,mu,rho,0.8)
# gd(N,V,mu,rho,1)
# gd(N,V,mu,rho,1.2)
# gd(N,V,mu,rho,1.4)


def gest(N,V,mu,rho,d):
    
    if d ==0:
        return g0(N,V,mu)
    else:
        return gd(N,V,mu,rho,d)
    
    
def gfuncest(N,V,mu,rho,ds):
    
    return np.array([gest(N,V,mu,rho,d) for d in ds])
    

# ### cluster example
    
# N = 10000

# scl = 4
# # V = np.array([[1,-0.9],[-0.9,1]])*scl
# V = np.array([[1,0.999],[0.999,1]])*scl
# # V = np.array([[1,0],[0,1]])*scl

# mn = -2
# mu = np.array([0,0])+mn

# rho = 5


# steps = np.arange(0,1.1,0.1)
# gs = gfuncest(N,V,mu,rho,steps)

# plt.plot(steps,gs[:,0])
# # plt.show()
# plt.plot(steps,gs[:,2])
# # plt.show()
# plt.plot(steps,gs[:,1], c="tab:orange")
# plt.plot(steps,gs[:,1], linestyle="dashed", c="tab:blue")
# plt.show()


    
    
## hate example
    
# N = 10000

# scl = 4
# V = np.array([[1,-0.9],[-0.9,1]])*scl
# # V = np.array([[1,0.9],[0.9,1]])*scl
# # V = np.array([[1,0],[0,1]])*scl

# mn = -2
# mu = np.array([0,0])+mn

# rho = 5


# steps = np.arange(0,1.1,0.1)
# gs = gfuncest(N,V,mu,rho,steps)

# plt.plot(steps,gs[:,0])
# # plt.show()
# plt.plot(steps,gs[:,2])
# # plt.show()
# plt.plot(steps,gs[:,1], c="tab:orange")
# plt.plot(steps,gs[:,1], linestyle="dashed", c="tab:blue")
# plt.show()
            
        
# ### neutral example
    
# N = 10000

# scl = 1
# # V = np.array([[1,-0.9],[-0.9,1]])*scl
# # V = np.array([[1,0.9],[0.9,1]])*scl
# V = np.array([[1,0],[0,1]])*scl

# mn = 0
# mu = np.array([0,0])+mn

# rho = 5


# steps = np.arange(0,1.1,0.1)
# gs = gfuncest(N,V,mu,rho,steps)

# plt.plot(steps,gs[:,0])
# # plt.show()
# plt.plot(steps,gs[:,2])
# # plt.show()
# plt.plot(steps,gs[:,1], c="tab:orange")
# plt.plot(steps,gs[:,1], linestyle="dashed", c="tab:blue")
# plt.show()


# ### mix and match example
    
# N = 10000

# scl = 10
# # V = np.array([[1,-0.9],[-0.9,1]])*scl
# # V = np.array([[1,0.9],[0.9,1]])*scl
# # V = np.array([[1,0],[0,1]])*scl
# V = np.array([[1,-4.5],[-4.5,25]])*scl

# mn = 0
# # mu = np.array([0,0])+mn
# mu = np.array([0,-2])+mn

# rho = 5


# steps = np.arange(0,1.1,0.1)
# gs = gfuncest(N,V,mu,rho,steps)

# plt.plot(steps,gs[:,0])
# # plt.show()
# plt.plot(steps,gs[:,2])
# # plt.show()
# plt.plot(steps,gs[:,1], c="tab:orange")
# plt.plot(steps,gs[:,1], linestyle="dashed", c="tab:blue")
# plt.show()


# ### dull example
    
# N = 10000

# scl = 1
# # V = np.array([[1,-0.9],[-0.9,1]])*scl
# V = np.array([[1,0.9],[0.9,1]])*scl
# # V = np.array([[1,0],[0,1]])*scl
# # V = np.array([[1,-4.5],[-4.5,25]])*scl

# mn = 0
# mu = np.array([2,2])+mn
# # mu = np.array([0,-2])+mn

# rho = 10


# steps = np.arange(0,1.1,0.1)
# gs = gfuncest(N,V,mu,rho,steps)

# plt.plot(steps,gs[:,0])
# # plt.show()
# plt.plot(steps,gs[:,2])
# # plt.show()
# plt.plot(steps,gs[:,1], c="tab:orange")
# plt.plot(steps,gs[:,1], linestyle="dashed", c="tab:blue")
# plt.show()

