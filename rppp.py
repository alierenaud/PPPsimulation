# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 12:42:08 2020

@author: alierenaud
"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

from rGP import GP
from rGP import expCov
from rGP import zeroMean
    
def isInUnitSquare(x):
    return(x[0]>0 and x[0]<1 and x[1]>0 and x[1]<1)

def expit(x):
    return(np.exp(x)/(1+np.exp(x)))

class PPP:  
    def __init__(self, loc):
        self.loc = loc

    @classmethod
    def randomHomog(cls,lam):
        N = int(random.poisson(lam=lam, size=1))
        loc = random.uniform(size=(N, 2))
        return cls(loc)
    
    @classmethod
    def randomNonHomog(cls,lam,fct):
        newPPP = cls.randomHomog(lam)
        index = np.greater(fct(newPPP.loc),random.uniform(size=newPPP.loc.shape[0]))
        newPPP.loc = newPPP.loc[index]
        return(newPPP)
    
    
    @classmethod
    def randomSGCD(cls,lam,tau,rho):
        newPPP = cls.randomHomog(lam)
        newGP = GP(zeroMean,expCov(1,1))
        marks = newGP.rGP(newPPP.loc)
        index = np.array(np.greater(expit(marks),random.uniform(size=marks.shape)))
        newPPP.loc = newPPP.loc[np.squeeze(index)]
        return(newPPP)
    

    
    @classmethod
    def randomHomogNS(cls,lam,alpha,sigma):
        newPPP = cls.randomHomog(lam)
        newLoc = np.empty(shape=(0,2))
        for i in newPPP.loc:
            N = random.poisson(alpha)
            jitter = random.normal(loc=0,scale=sigma,size=(N,2))
            newLoc = np.concatenate((newLoc,i+jitter))
        newPPP.loc = newLoc[[isInUnitSquare(x) for x in newLoc]]
        return(newPPP)
    
    
    @classmethod
    def randomNonHomogNS(cls,lam,fct,alpha,sigma):
        newPPP = cls.randomNonHomog(lam,fct)
        newLoc = np.empty(shape=(0,2))
        for i in newPPP.loc:
            N = random.poisson(alpha)
            jitter = random.normal(loc=0,scale=sigma,size=(N,2))
            newLoc = np.concatenate((newLoc,i+jitter))
        newPPP.loc = newLoc[[isInUnitSquare(x) for x in newLoc]]
        return(newPPP)
    
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')

        plt.plot(self.loc[:,0],self.loc[:,1], 'o')
        plt.xlim(0,1)
        plt.ylim(0,1)

        plt.show()
        

class mtPPP:  
    def __init__(self, pps):
        self.pps = pps
        n = [x.loc.shape[0] for x in pps]
        K = pps.shape[0]
        c = np.cumsum(n)
        self.nObs = sum(n)
        self.typeMatrix = np.zeros(shape=(K,self.nObs))
        self.typeMatrix[0,0:c[0]] = 1
        for i in list(range(1,K)):
            self.typeMatrix[i,c[i-1]:c[i]] = 1
            
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')

        for pp in self.pps:
            plt.plot(pp.loc[:,0],pp.loc[:,1], 'o')
        
        plt.xlim(0,1)
        plt.ylim(0,1)

        plt.show()









