# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:58:16 2021

@author: alier
"""

import numpy as np
import matplotlib.pyplot as plt
from rGP import GP
from rGP import gaussianCov




### usage examples

### zero mean

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


res = 25
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

### zero mean (cond)

nlocObs = 100
nloc = res**2

locObs = gridLoc[0:nlocObs,:]
locPred = gridLoc[nlocObs:nloc,:]


valObs = newGP.rGP(gridLoc[0:nlocObs,:])

resCondGP = newGP.rCondGP(locPred, locObs, valObs)

resGP = np.concatenate((valObs, resCondGP))

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

### cross mean

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

### cross mean (cond)

nlocObs = 500
nloc = res**2

locObs = gridLoc[0:nlocObs,:]
locPred = gridLoc[nlocObs:nloc,:]


valObs = newGP.rGP(gridLoc[0:nlocObs,:])

resCondGP = newGP.rCondGP(locPred, locObs, valObs)

resGP = np.concatenate((valObs, resCondGP))

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

### circle mean

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

### circle mean (cond)

nlocObs = 500
nloc = res**2

locObs = gridLoc[0:nlocObs,:]
locPred = gridLoc[nlocObs:nloc,:]


valObs = newGP.rGP(gridLoc[0:nlocObs,:])

resCondGP = newGP.rCondGP(locPred, locObs, valObs)

resGP = np.concatenate((valObs, resCondGP))

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












