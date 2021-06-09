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
def fct(x):
    return(np.exp(-(0.25-np.sqrt((x[:,0]-0.5)**2+(x[:,1]-0.5)**2))**2/0.003)*0.8+0.10)
# def fct(x):
#     return(np.exp(-np.minimum((x[:,0]-0.5)**2,(x[:,1]-0.5)**2)/0.007)*0.8+0.10)

lam_sim=500

pointpo = PPP.randomNonHomog(lam_sim,fct)
pointpo.plot()


newGP = GP(zeroMean,gaussianCov(2,0.5))

niter=1000

import time

t0 = time.time()
locations,values,lams = MCMCadams(niter,100,newGP,pointpo,20,10,0.1,10,lam_sim,1000)
t1 = time.time()

total1 = t1-t0


# ### do some plot

# import matplotlib.pyplot as plt


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_aspect('equal')

# plt.plot(locations.obsLoc()[:,0],locations.obsLoc()[:,1], 'o')
# plt.plot(locations.thinLoc()[:,0],locations.thinLoc()[:,1], 'o')
# plt.xlim(0,1)
# plt.ylim(0,1)

# plt.show()



# i=0

# while i<niter:


#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_aspect('equal')
    
#     plt.plot(locations.obsLoc()[:,0],locations.obsLoc()[:,1], 'o')
#     locations.sampNb=i
#     plt.plot(locations.thinLoc()[:,0],locations.thinLoc()[:,1], 'o')
#     plt.xlim(0,1)
#     plt.ylim(0,1)
    
#     plt.show()
    
#     i+=1



# plt.plot(lams)




# def makeGrid(xlim,ylim,res):
#     grid = np.ndarray((res**2,2))
#     xlo = xlim[0]
#     xhi = xlim[1]
#     xrange = xhi - xlo
#     ylo = ylim[0]
#     yhi = ylim[1]
#     yrange = yhi - ylo
#     xs = np.arange(xlo, xhi, step=xrange/res) + xrange/res*0.5
#     ys = np.arange(ylo, yhi, step=yrange/res) + yrange/res*0.5
#     i=0
#     for x in xs:
#         j=0
#         for y in ys:
#             grid[i*res+j,:] = [x,y]
#             j+=1
#         i+=1
#     return(grid)


# res = 40
# gridLoc = makeGrid([0,1], [0,1], res)


# def expit(x):
#     return(np.exp(x)/(1+np.exp(x)))


# resGP = np.empty(shape=niter,dtype=np.ndarray)
# i=0
# t0 = time.time()
# while(i < niter):
#     locations.sampNb=i
#     values.sampNb=i
#     resGP[i] = lams[i]*expit(newGP.rCondGP(gridLoc,locations.totLoc(),values.totLoc()))
#     print(i)
#     i+=1
# t1 = time.time()

# total2 = t1-t0


# ### plot mean intensity

# imGP = np.transpose(np.mean(resGP[100:]).reshape(res,res))
    
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




# i=0
# while(i < niter):


    
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






