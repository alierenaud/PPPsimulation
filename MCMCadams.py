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
# inserts or deletes a thinned event along with the GP value
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
    

# ###
# # beta random variable alpha=mu*phi, beta=(1-mu)*phi
# ###
    
# def rbeta(mu,phi):
#     return(beta.rvs(mu*phi,(1-mu)*phi))
        
# ### TESTER: rbeta    
# rbeta(0.8, 1)
    
###
# jitter a point x in [0,1] according to a beta(x*phi,(1-x)*phi)
# phi = kappa/min(x,1-x), kappa > 1
###

def jitterBeta(x,kappa):
    phi = kappa/min(x,1-x)
    return(beta.rvs(x*phi,(1-x)*phi))

### TESTER: jitterBeta
jitterBeta(0.5,10)
jitterBeta(0.5,3)
jitterBeta(0.9,10)


###
# kernel of the beta(kappa) jitter from (2D) xOld to xNew
###

def kernelBeta(xNew,xOld,kappa):
    phi0 = kappa/min(xOld[:,0],1-xOld[:,0])
    phi1 = kappa/min(xOld[:,1],1-xOld[:,1])
    d0 = beta.pdf(xNew[:,0], xOld[:,0]*phi0, (1-xOld[:,0])*phi0)
    d1 = beta.pdf(xNew[:,1], xOld[:,1]*phi1, (1-xOld[:,1])*phi1)
    return(d0*d1)

### TESTER: kernelBeta
xOld = np.array([[0.5,0.5]])
xNew = np.array([[0.6,0.6]])
kappa = 10

kernelBeta(xNew, xOld, kappa)


xOld = np.array([[0.5,0.5]])
xNew = np.array([[0.7,0.7]])
kappa = 10

kernelBeta(xNew, xOld, kappa)


xOld = np.array([[0.5,0.5]])
xNew = np.array([[0.4,0.4]])
kappa = 10

kernelBeta(xNew, xOld, kappa)


xOld = np.array([[0.7,0.7]])
xNew = np.array([[0.8,0.8]])
kappa = 10

kernelBeta(xNew, xOld, kappa)

xOld = np.array([[0.7,0.7]])
xNew = np.array([[0.9,0.9]])
kappa = 10

kernelBeta(xNew, xOld, kappa)


###
# loops through thinned points and proposes new locations and GP values
###

def locationSampler(kappa, thisGP, locThin, valThin, locObs, valObs):
    
    i=0
    for thisLoc in locThin:
        thisLoc = np.array([thisLoc])
        newLoc = np.array([[jitterBeta(thisLoc[:,0],kappa),jitterBeta(thisLoc[:,1],kappa)]])
        
        locTot = np.concatenate((locThin, locObs))
        valTot = np.concatenate((valThin, valObs))
        newVal = thisGP.rCondGP(newLoc, locTot, valTot)
        
        acc_loc = kernelBeta(thisLoc, newLoc, kappa)/(1+np.exp(newVal))*(1+np.exp(valThin[i]))/kernelBeta(newLoc, thisLoc, kappa) 
        
        U = random.uniform(size=1)
        
        if U < acc_loc:
            locThin[i] = newLoc
            valThin[i] = newVal
        
        i+=1
    return(locThin, valThin)

### TESTER locationSampler
kappa=10
newGP = GP(zeroMean,gaussianCov(1,1))

lam=10
pointpo = PPP.randomHomog(lam)

locThin = pointpo.loc
valThin = newGP.rGP(pointpo.loc) 

pointpo = PPP.randomHomog(lam)

locObs = pointpo.loc
valObs = newGP.rCondGP(locObs, locThin, valThin) 

    
locThin, valThin = locationSampler(kappa, newGP, locThin, valThin, locObs, valObs)    
locThin, valThin 








