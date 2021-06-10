# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:31:35 2021

@author: alier
"""

from MCMCadams import MCMCadams
import numpy as np
from rppp import PPP
from rGP import GP
from rGP import gaussianCov 
from rGP import zeroMean



# def fct(x):
#     return(np.exp((-x[:,0]**2-x[:,1]**2)/0.3))
# def fct(x):
#     return(np.exp(-(0.25-np.sqrt((x[:,0]-0.5)**2+(x[:,1]-0.5)**2))**2/0.003)*0.8+0.10)
def fct(x):
    return(np.exp(-np.minimum((x[:,0]-0.5)**2,(x[:,1]-0.5)**2)/0.007)*0.8+0.10)

lam_sim=1000

pointpo = PPP.randomNonHomog(lam_sim,fct)
pointpo.plot()


newGP = GP(zeroMean,gaussianCov(2,0.1))

niter=1000
nInsDelMov = 100

import time

t0 = time.time()
locations,values,lams = MCMCadams(niter,100,newGP,pointpo,nInsDelMov,10,0.1,10,lam_sim,1000,p=True)
t1 = time.time()

total1 = t1-t0


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


def expit(x):
    return(np.exp(x)/(1+np.exp(x)))


import matplotlib.pyplot as plt

resGP = np.empty(shape=(niter-1,res**2,1),dtype=np.ndarray)
# meanGP = np.zeros(shape=(res**2,1))
i=0
t0 = time.time()
while(i < niter-1):
    locations = np.loadtxt("locations"+str(i)+".csv", delimiter=",")
    values = np.loadtxt("values"+str(i)+".csv", delimiter=",")
    # np.savetxt("resGP"+str(i)+".csv",lams[i+1]*expit(newGP.rCondGP(gridLoc,locations,np.transpose([values]))) ,delimiter=",")
    resGP[i] = lams[i+1]*expit(newGP.rCondGP(gridLoc,locations,np.transpose([values])))
    # meanGP = ((i+1)*meanGP + lams[i+1]*expit(newGP.rCondGP(gridLoc,locations,np.transpose([values]))))/(i+2)
    
    # imGP = np.transpose(resGP.reshape(res,res))
    
    # x = np.linspace(0,1, res+1) 
    # y = np.linspace(0,1, res+1) 
    # X, Y = np.meshgrid(x,y) 
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_aspect('equal')
    
    # plt.pcolormesh(X,Y,imGP, cmap='cool')
    
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    # plt.colorbar()
    # plt.scatter(pointpo.loc[:,0],pointpo.loc[:,1], color= "black", s=1)
    
    # plt.show()
    
    print(i)
    i+=1
t1 = time.time()

total2 = t1-t0

### do some plot




fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.plot(locations.obsLoc()[:,0],locations.obsLoc()[:,1], 'o')
plt.plot(locations.thinLoc()[:,0],locations.thinLoc()[:,1], 'o')
plt.xlim(0,1)
plt.ylim(0,1)

plt.show()

nObs = pointpo.loc.shape[0]

i=0

while i<niter-1:

    locations = np.loadtxt("locations"+str(i)+".csv", delimiter=",")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    
    plt.plot(locations[:nObs,0],locations[:nObs,1], 'o')

    plt.plot(locations[nObs:,0],locations[nObs:,1], 'o')
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    plt.show()
    print(i)
    i+=1



# # plt.plot(lams)





meanGP = np.mean(resGP, axis=0, dtype=np.float32)

### plot mean intensity

imGP = np.transpose(meanGP.reshape(res,res))
    
x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
    
plt.pcolormesh(X,Y,imGP, cmap='cool')
    
plt.xlim(0,1)
plt.ylim(0,1)
plt.colorbar()
plt.scatter(pointpo.loc[:,0],pointpo.loc[:,1], color= "black", s=1)
    
plt.show()




# i=0
# while(i < niter-1):


    
#     imGP = np.transpose(resGP[i].reshape(res,res))
    
#     x = np.linspace(0,1, res+1) 
#     y = np.linspace(0,1, res+1) 
#     X, Y = np.meshgrid(x,y) 
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_aspect('equal')
    
#     plt.pcolormesh(X,Y,imGP, cmap='cool')
    
#     plt.xlim(0,1)
#     plt.ylim(0,1)
#     plt.colorbar()
#     plt.scatter(pointpo.loc[:,0],pointpo.loc[:,1], color= "black", s=1)
    
#     plt.show()
#     # fig.savefig("Int"+str(i)+".pdf", bbox_inches='tight')
#     i+=1






