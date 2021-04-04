# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:12:04 2021

@author: alier
"""

from rppp import PPP
import numpy as np

from numpy import random
import matplotlib.pyplot as plt

### usage example

lam=500
alpha = 2.2
sigma = 0.02

pointpo = PPP.randomHomog(lam)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.plot(pointpo.loc[:,0],pointpo.loc[:,1], 'o')
plt.xlim(0,1)
plt.ylim(0,1)

plt.show()

fig.savefig("foo.pdf", bbox_inches='tight')   

# def fct(x):
#     return(np.exp(-np.minimum((x[:,0]-0.5)**2,(x[:,1]-0.5)**2)/0.003))

# def fct(x):
#     return(np.exp(-((x[:,0])+(x[:,1]))**2/0.3))
def fct(x):
    return(np.exp(-(0.25-np.sqrt((x[:,0]-0.5)**2+(x[:,1]-0.5)**2))**2/0.003))

index = np.greater(fct(pointpo.loc),random.uniform(size=pointpo.loc.shape[0]))
nhPPP = pointpo.loc[index]



fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.plot(pointpo.loc[:,0],pointpo.loc[:,1], 'o')
plt.plot(nhPPP[:,0],nhPPP[:,1], 'o')
plt.xlim(0,1)
plt.ylim(0,1)

plt.show()
fig.savefig("foo1.pdf", bbox_inches='tight')  

def isInUnitSquare(x):
    return(x[0]>0 and x[0]<1 and x[1]>0 and x[1]<1)


nsLoc = np.empty(shape=(0,2))
allocs = np.empty(shape=(0), dtype=int)
j=0
for i in nhPPP:
    N = random.poisson(alpha)
    jitter = random.normal(loc=0,scale=sigma,size=(N,2))
    nsLoc = np.concatenate((nsLoc,i+jitter))
    allocs = np.concatenate((allocs,np.full(N, j)))
    j+=1
index = [isInUnitSquare(x) for x in nsLoc]
nsLoc = nsLoc[index]
allocs = allocs[index]        
        
        
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.plot(pointpo.loc[:,0],pointpo.loc[:,1], 'o', c="#00743F", alpha=0.5)
plt.plot(nhPPP[:,0],nhPPP[:,1], 'o', c="#1E65A7")
plt.plot(nsLoc[:,0],nsLoc[:,1], 'ro', ms=3)

N = allocs.shape[0] 
k=0
while(k<N):
    plt.plot([nsLoc[k,0],nhPPP[allocs[k],0]], [nsLoc[k,1],nhPPP[allocs[k],1]],"r-", lw=0.3)
    k +=1
    
plt.plot([1/4,1/4], [1/4,3/4],c="black", lw=1)   
plt.plot([3/4,1/4], [1/4,1/4],c="black", lw=1)
plt.plot([3/4,3/4], [3/4,1/4],c="black", lw=1)
plt.plot([1/4,3/4], [3/4,3/4],c="black", lw=1)


plt.xlim(0,1)
plt.ylim(0,1)

plt.show()     
        
fig.savefig("foo2.pdf", bbox_inches='tight')     
        
        


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.plot(pointpo.loc[:,0],pointpo.loc[:,1], 'o', c="#00743F", alpha=0.5)
plt.plot(nhPPP[:,0],nhPPP[:,1], 'o', c="#1E65A7")
plt.plot(nsLoc[:,0],nsLoc[:,1], 'ro', ms=3)

N = allocs.shape[0] 
k=0
while(k<N):
    plt.plot([nsLoc[k,0],nhPPP[allocs[k],0]], [nsLoc[k,1],nhPPP[allocs[k],1]],"r-", lw=0.3)
    k +=1



plt.xlim(1/4,3/4)
plt.ylim(1/4,3/4)

plt.show()     
        
fig.savefig("foo3.pdf", bbox_inches='tight')  











        
        
        