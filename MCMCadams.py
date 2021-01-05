# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:51:32 2021

@author: alier
"""

import numpy as np
from numpy import random
from scipy.stats import beta
from rGP import GP
from rGP import gaussianCov
from rppp import PPP
import matplotlib.pyplot as plt



# Source: (Adams, 2009) Tractable Nonparametric Bayesian Inference in Poisson Processes
# with Gaussian Process Intensities



### sampling number of thinned events (3.3) ###

### 
# Takes the location of all points and returns a new one drawn uniformly 
# along with the its conditionnal GP value
###

def insProp(thisGP, locObs, valObs):
    
    newLoc =  random.uniform(size=(1, 2))
    
    valNewLoc = thisGP.rCondGP(newLoc, locObs, valObs)
    
    return(newLoc, valNewLoc)



### TESTER: insProp
def zeroMean(x):
    return(0) 

newGP = GP(zeroMean,gaussianCov(1,1))

lam=100
pointpo = PPP.randomHomog(lam)

resGP = newGP.rGP(pointpo.loc)

insProp(newGP, pointpo.loc, resGP)
###

### 
# Takes the location of thinned points and removes 1 uniformly
###

def delProp(locThin, valThin):
    
    nlocThin = locThin.shape[0]
    
    delInd = random.choice(np.array(range(0,nlocThin)))
    
    oldVal = valThin[delInd]
    
    newLocThin = np.delete(locThin, delInd, 0)
    newValThin = np.delete(valThin, delInd, 0)
    
    return(oldVal,newLocThin,newValThin)

### TESTER: delProp
newGP = GP(zeroMean,gaussianCov(1,1))

lam=10
pointpo = PPP.randomHomog(lam)

resGP = newGP.rGP(pointpo.loc)

delProp(pointpo.loc, resGP)
###

###
# take number of thin events and return the prob of insertion
###

def b(nthin):
    if nthin==0:
        return(1)
    else:
        return(0.5)

### TESTER: b
b(0)
b(random.choice(np.array(range(0,5))))

###
# inserts or deletes a thinned events along with the GP value
###

def nthinSampler(lam, thisGP, locThin, valThin, locObs, valObs):
    
    nthin = locThin.shape[0]
    
    B = random.binomial(1,b(nthin),1)
    
    if B:
        locTot = np.concatenate((locThin, locObs))
        valTot = np.concatenate((valThin, valObs))
        
        newLoc, newVal = insProp(thisGP, locTot, valTot)
        
        acc_ins = (1-b(nthin+1))/b(nthin)*lam/(nthin+1)/(1+np.exp(newVal))
        
        U = random.uniform(size=1)
        
        if U < acc_ins:
            locThin = np.concatenate((locThin, newLoc))
            valThin = np.concatenate((valThin, newVal))
    else:
        oldVal, newThinLocs, newThinVal = delProp(locThin, valThin)
        
        acc_del = b(nthin-1)/(1-b(nthin))*nthin/lam*(1+np.exp(oldVal))
                                                  
        U = random.uniform(size=1)
        
        if U < acc_del:
            locThin = newThinLocs
            valThin = newThinVal

    return(locThin, valThin)
    
### TESTER: nthinSampler
newGP = GP(zeroMean,gaussianCov(1,1))

lam=10
pointpo = PPP.randomHomog(lam)

resGP = newGP.rGP(pointpo.loc) 

locThin = np.empty((0,2))
valThin = np.empty((0,1))

    
locThin, valThin = nthinSampler(10, newGP, locThin, valThin, pointpo.loc, resGP)    
locThin, valThin 


# ###
# # checks if 2D coordinate fall in unit square
# ###

# def isInUnitSquare(x):
#     return(x[0]>0 and x[0]<1 and x[1]>0 and x[1]<1)
    
# ### TESTER: isInUnitSquare

# x = np.array([-2,3])
# isInUnitSquare(x)
    
# x = np.array([0.5,0.7])
# isInUnitSquare(x)

# x = np.array([2,0.7])
# isInUnitSquare(x)

# x = np.array([0.4,-1])
# isInUnitSquare(x)
    

###
# beta random variable alpha=mu*phi, beta=(1-mu)*phi
###
    
def rbeta(mu,phi):
    return(beta.rvs(mu*phi,(1-mu)*phi))
        
### TESTER: rbeta    
rbeta(0.8, 1)
    


### to check    
mu = np.random.uniform()
kappa = 10
phi = kappa/min(mu,1-mu)




x = np.linspace(0,1,100)
plt.plot(x, beta.pdf(x, mu*phi, (1-mu)*phi),
       'r-', lw=5, alpha=0.6, label='beta pdf')
    
###






