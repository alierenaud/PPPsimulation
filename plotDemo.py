# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:12:04 2021

@author: alier
"""

from rppp import PPP
import numpy as np

from numpy import random
import matplotlib.pyplot as plt

from rGP import GP
from rGP import gaussianCov
from rGP import expCov
from rGP import zeroMean

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



### SGCD with thinned and GP at grid locations


lam=100
tau=1/3
rho=1

def expit(x):
    return(np.exp(x)/(1+np.exp(x)))

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


res = 50
gridLoc = makeGrid([0,1], [0,1], res)


locs = PPP.randomHomog(lam).loc

newGP = GP(zeroMean,expCov(tau,rho))
GPvals = newGP.rGP(np.concatenate((locs,gridLoc)))


gridInt = lam*expit(GPvals[locs.shape[0]:,:])


locProb  = expit(GPvals[:locs.shape[0],:])
index = np.array(np.greater(locProb,random.uniform(size=locProb.shape)))
locObs = locs[np.squeeze(index)]
locThin = locs[np.logical_not(np.squeeze(index))]




### cox process


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.xlim(0,1)
plt.ylim(0,1)

plt.scatter(locObs[:,0],locObs[:,1])

plt.show()

### int + obs + thin

imGP = np.transpose(gridInt.reshape(res,res))

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.pcolormesh(X,Y,imGP, cmap='winter')

plt.xlim(0,1)
plt.ylim(0,1)
plt.colorbar()

plt.scatter(locObs[:,0],locObs[:,1], color= "black", s=1)
plt.scatter(locThin[:,0],locThin[:,1], color= "white", s=1)


plt.show()


#### multitype SGCP

from scipy.stats import matrix_normal
# from scipy.stats import multinomial

lam=400
tau=1
rho=2

locs = PPP.randomHomog(lam).loc


newGP = GP(zeroMean,expCov(tau,rho))

U = newGP.covMatrix(locs)

n=50
eps=1/n
X = np.array([[0,eps/np.sqrt(1+eps**2),1/np.sqrt(1+eps**2)],[0,-eps/np.sqrt(1+eps**2),1/np.sqrt(1+eps**2)],[1/np.sqrt(2),0,-1/np.sqrt(2)]])
V = X@np.transpose(X)



# V = np.array([[1,0.9999],[0.9999,1]]) 



X = matrix_normal.rvs(rowcov=U, colcov=V)

def multExpit(x):
    N = np.sum(np.exp(x))
    probs = np.array([np.exp(i)/(1+N) for i in x])
    return(np.append(probs,1-np.sum(probs)))
        
        
probs = np.array([multExpit(x) for x in X])

nch = probs.shape[1]

colours = np.array([np.random.choice(nch,p=p) for p in probs])

locs0 = locs[colours == 0]
locs1 = locs[colours == 1]
locs2 = locs[colours == 2]
locs3 = locs[colours == 3]


### plot of real for multivariate SGCP

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')


plt.xlim(0,1)
plt.ylim(0,1)

plt.scatter(locs0[:,0],locs0[:,1],label="Oak")
plt.scatter(locs1[:,0],locs1[:,1],label="Pine")
plt.scatter(locs2[:,0],locs2[:,1],label="Maple")
# plt.scatter(locs3[:,0],locs3[:,1], c="grey")
plt.legend(bbox_to_anchor=(1, 0.8))

plt.show()


### plot of real for multivariate SGCP

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')


plt.xlim(0,1)
plt.ylim(0,1)

plt.scatter(locs0[:,0],locs0[:,1],label="Oak")
plt.scatter(locs1[:,0],locs1[:,1],label="Pine")
plt.scatter(locs2[:,0],locs2[:,1],label="Maple")
plt.scatter(locs3[:,0],locs3[:,1], c="grey",label="Thinned")


plt.legend(bbox_to_anchor=(1, 0.8))

plt.show()



        
        
        