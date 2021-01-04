# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:51:32 2021

@author: alier
"""

import numpy as np
from numpy import random
from rGP import GP
from rGP import gaussianCov
from rppp import PPP


# Source: (Adams, 2009) Tractable Nonparametric Bayesian Inference in Poisson Processes
# with Gaussian Process Intensities



### sampling number of thinned events (3.3)

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

def delProp(locObs):
    
    nloc = locObs.shape[0]
    
    delInd = random.choice(np.array(range(0,nloc)))
    
    newLocObs = np.delete(locObs, delInd, 0)
    
    return(newLocObs)

### TESTER: delProp
lam=10
pointpo = PPP.randomHomog(lam)

delProp(pointpo.loc)
###









