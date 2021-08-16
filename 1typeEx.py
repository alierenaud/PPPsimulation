# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:40:04 2021

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





######
### Examples and testing of the multiType SGCD MCMC alogrithm
######

## Generate a mtype SGCD

lam=500
# rho=2

### using function

def fct(x):
      return(np.exp((-x[:,0]**2-x[:,1]**2)/0.3))
# def fct(x):
#     return(1-(np.exp(-np.minimum((x[:,0]-0.5)**2,(x[:,1]-0.5)**2)/0.007)))

pointpo = PPP.randomNonHomog(lam,fct)

pointpo.plot()

###




mtpp = mtPPP(np.array([pointpo]))

mtpp.plot()



### initialize other MCMC parameters

size=1000
nInsDelMov = lam//10




K = mtpp.K

n=lam*2

# V_mc=np.linalg.inv(V)/n
V_mc=np.identity(K)/n

# T_init = np.linalg.inv(V)
T_init=np.identity(K) 


kappa=10
delta=0.1
L=10
mu=lam
sigma2=100
a=200
b=100

p=True


import time

t0 = time.time()
locations,values,lams,rhos,Ts = MCMCadams(size,lam,a/b,T_init,mtpp,nInsDelMov,kappa,delta,L,mu,sigma2,p,a,b,n,V_mc)
t1 = time.time()

tt1 = t1-t0

### Diagnostics




### trace plots

plt.plot(lams)
plt.show()

plt.plot(rhos)
plt.show()

plt.plot(Ts[:,0,0])
plt.show()



### GP values trace

nObs = mtpp.nObs


obsGP = np.empty(shape=(size-1,nObs,K))

i=0
while(i < size-1):
    values = np.loadtxt("values"+str(i)+".csv", delimiter=",")
    # np.savetxt("resGP"+str(i)+".csv",lams[i+1]*expit(newGP.rCondGP(gridLoc,locations,np.transpose([values]))) ,delimiter=",")

    obsGP[i] = np.transpose([values[0:nObs]])

    
    print(i)
    i+=1
    
    
fig, axs = plt.subplots(nObs,figsize=(10,nObs))    
    
i=0
while(i < nObs):
    axs[i].plot(obsGP[:,i])

    
    print(i)
    i+=1    

# plt.show()
fig.savefig("0GPtraces.pdf", bbox_inches='tight')


### dancing thinned locations
from matplotlib.backends.backend_pdf import PdfPages
mpdf = PdfPages('0thinLocs.pdf')


i=0
while(i < size-1):
    locations = np.loadtxt("locations"+str(i)+".csv", delimiter=",")
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    for pp in mtpp.pps:
            plt.plot(pp.loc[:,0],pp.loc[:,1], 'o')
    plt.plot(locations[nObs:,0],locations[nObs:,1], 'o', c="grey")
    plt.xlim(0,1)
    plt.ylim(0,1)

    # plt.show()
    mpdf.savefig(bbox_inches='tight') 
    plt.close(fig)
    
    
    print(i)
    i+=1

mpdf.close()


### type intensities

def makeGrid(xlim,ylim,res):
    grid = np.ndarray((res**2,2))
    xlo = xlim[0]
    xhi = xlim[1]
    xrange = xhi - xlo
    ylo = ylim[0]
    yhi = ylim[1]
    yrange = yhi - ylo
    xs = np.arange(xlo, xhi, step=xrange/res) + xrange/res*0.5
    ys = np.arange(ylo, yhi, step=yrange/res) + yrange/res*0.5
    i=0
    for x in xs:
        j=0
        for y in ys:
            grid[i*res+j,:] = [x,y]
            j+=1
        i+=1
    return(grid)


res = 20
gridLoc = makeGrid([0,1], [0,1], res)

def multExpit(x):
    N = np.sum(np.exp(x))
    probs = np.array([np.exp(i)/(1+N) for i in x])
    return(np.append(probs,1-np.sum(probs)))

resGP = np.empty(shape=(size,res**2,K+1))
# meanGP = np.zeros(shape=(res**2,1))
i=0
j=0
t0 = time.time()
while(i < size-1):
    locations = np.loadtxt("locations"+str(i)+".csv", delimiter=",")
    values = np.loadtxt("values"+str(i)+".csv", delimiter=",")
    # np.savetxt("resGP"+str(i)+".csv",lams[i+1]*expit(newGP.rCondGP(gridLoc,locations,np.transpose([values]))) ,delimiter=",")
    newGP = GP(zeroMean,expCov(1,rhos[i+1]))
    
    ## propose new value from MGP(.|totVal)
    
    s_11 = newGP.cov(gridLoc,gridLoc)
    S_21 = newGP.cov(locations,gridLoc)
    S_22 = newGP.cov(locations,locations)
    
    S_12S_22m1 = np.dot(np.transpose(S_21),np.linalg.inv(S_22))
    
    mu = np.dot(S_12S_22m1,np.transpose([values]))
    spatSig = s_11 - np.dot(S_12S_22m1,S_21)
    
    A = np.linalg.cholesky(Ts[i+1])
    
    K = A.shape[0]
    Am = sp.linalg.solve_triangular(A,np.identity(K),lower=True)
    
    newVal = np.linalg.cholesky(spatSig)@np.random.normal(size=(res**2,K))@Am+mu
    
    
    resGP[j] = lams[i+1]*np.array([multExpit(val) for val in newVal])

    
    print(i)
    i+=1
    j+=1
t1 = time.time()

total2 = t1-t0


meanGP = np.mean(resGP, axis=0, dtype=np.float32)

maxi = np.max(meanGP)
mini = np.min(meanGP)

### plot mean intensity type 1

imGP = np.transpose(meanGP[:,0].reshape(res,res))
    
x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
    
plt.pcolormesh(X,Y,imGP, cmap='gray', vmin=mini, vmax=maxi)

for pp in mtpp.pps:
    plt.plot(pp.loc[:,0],pp.loc[:,1], 'o')
    
plt.xlim(0,1)
plt.ylim(0,1)
plt.colorbar()
# plt.scatter(pointpo.loc[:,0],pointpo.loc[:,1], color= "black", s=1)
    
plt.show()
# fig.savefig("Int1.pdf", bbox_inches='tight')




    
    
    
### plot mean intensity thinned

imGP = np.transpose(meanGP[:,1].reshape(res,res))
    
x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
    
plt.pcolormesh(X,Y,imGP, cmap='gray',vmin=mini, vmax=maxi)

for pp in mtpp.pps:
    plt.plot(pp.loc[:,0],pp.loc[:,1], 'o')
    
plt.xlim(0,1)
plt.ylim(0,1)
plt.colorbar()
# plt.scatter(pointpo.loc[:,0],pointpo.loc[:,1], color= "black", s=1)
    
plt.show()
# fig.savefig("thinInt.pdf", bbox_inches='tight')



