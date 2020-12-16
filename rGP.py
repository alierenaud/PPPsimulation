# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 12:42:08 2020

@author: alierenaud
"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import numpy.linalg
import numpy.matlib




### gaussian process

class GP:  
    
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        
    
    def meanVec(self, loc):
        nloc = loc.shape[0]
        mu = np.matlib.zeros((nloc,1))
        i=0
        for x in loc:
            mu[i,0] = self.mean(x)
            i+=1
        return(mu)
    
    
    def covMatrix(self, loc):
        nloc = loc.shape[0]
        Sigma = np.matlib.zeros((nloc,nloc))
        i=0
        for x in loc:
            j=0
            for y in loc:
                Sigma[i,j] = self.cov(x,y)
                j+=1
            i+=1
        return(Sigma)
    
    def rGP(self, loc):
        Sigma = self.covMatrix(loc)
        nloc = loc.shape[0]
        Z = random.normal(size=(nloc,1))
        L = np.linalg.cholesky(Sigma)
        return(np.matmul(L,Z)+self.meanVec(loc))
    
    
def gaussianCov(sigma2,l):
    def evalCov(x,y):
        return(sigma2*np.exp(-np.linalg.norm(x-y)/2/l**2))
    return(evalCov)  

def zeroMean(x):
    return(0)    
    

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


res = 100
gridLoc = makeGrid([0,1], [0,1], res)


    


newGP = GP(zeroMean,gaussianCov(1,1))

resGP = newGP.rGP(gridLoc)

#### to make plot ####

imGP = resGP.reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.pcolormesh(X,Y,imGP)

plt.xlim(0,1)
plt.ylim(0,1)


plt.show()

#### ####


def fct(x):
    return(np.exp(-np.minimum((x[0]-0.5)**2,(x[1]-0.5)**2)/0.003))

newGP = GP(fct,gaussianCov(1,1))

resGP = newGP.rGP(gridLoc)

#### to make plot ####

imGP = resGP.reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.pcolormesh(X,Y,imGP)

plt.xlim(0,1)
plt.ylim(0,1)


plt.show()

#### ####


def fct(x):
    return(np.exp(-(0.25-np.sqrt((x[0]-0.5)**2+(x[1]-0.5)**2))**2/0.003))

newGP = GP(fct,gaussianCov(1,1))

resGP = newGP.rGP(gridLoc)

#### to make plot ####

imGP = resGP.reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.pcolormesh(X,Y,imGP)

plt.xlim(0,1)
plt.ylim(0,1)


plt.show()

#### ####












