# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 12:42:08 2020

@author: alierenaud
"""

import numpy as np
from numpy import random
import numpy.linalg
import numpy.matlib
from scipy.spatial import distance_matrix




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
        return(self.cov(loc,loc))
    
    def rGP(self, loc):
        Sigma = self.covMatrix(loc)
        nloc = loc.shape[0]
        Z = random.normal(size=(nloc,1))
        L = np.linalg.cholesky(Sigma)
        return(np.matmul(L,Z)+self.meanVec(loc))
    
    def rCondGP(self, locPred, locObs, valObs):
        loc = np.concatenate((locPred, locObs))
        Sigma = self.covMatrix(loc)
        mu = self.meanVec(loc)
        
        nlocPred = locPred.shape[0]
        nlocObs = locObs.shape[0]
        nloc = nlocPred + nlocObs
        
        mu1 = mu[0:nlocPred]
        mu2 = mu[nlocPred:nloc]
        
        Sigma11 = Sigma[0:nlocPred, 0:nlocPred]
        Sigma12 = Sigma[0:nlocPred, nlocPred:nloc]
        Sigma21 = np.transpose(Sigma12)
        Sigma22 = Sigma[nlocPred:nloc, nlocPred:nloc]
        
        invSigma22 = np.linalg.inv(Sigma22)
        
        mu1c2 = mu1 + np.matmul(np.matmul(Sigma12,invSigma22),  valObs-mu2)
        Sigma1c2 = Sigma11 - np.matmul(np.matmul(Sigma12,invSigma22),  Sigma21)
        
        L = np.linalg.cholesky(Sigma1c2)
        Z = random.normal(size=(nlocPred,1))
        return(np.matmul(L,Z)+mu1c2)
    
    def rCondGP1DSigma(self, locPred, locObs, valObs, Sigma):

        # mu = self.meanVec(loc)
        

        # nlocObs = locObs.shape[0]
        
        # mu1 = mu[0:nlocPred]
        # mu2 = mu[nlocPred:nloc]
        
        Sigma11 = self.cov(locPred,locPred)
        Sigma12 =  self.cov(locPred,locObs)
        Sigma21 = np.transpose(Sigma12)
        Sigma22 = Sigma
        
        newSigma = np.block([[Sigma11,Sigma12],[Sigma21,Sigma22]])
        
        invSigma22 = np.linalg.inv(Sigma22)
        
        mu1c2 = Sigma12@invSigma22@valObs
        Sigma1c2 = Sigma11 - Sigma12@invSigma22@Sigma21
        
        L = np.linalg.cholesky(Sigma1c2)
        Z = random.normal(size=(1,1))
        return(np.matmul(L,Z)+mu1c2, newSigma)

    
    
def gaussianCov(sigma2,l):
    def evalCov(x,y):
        return(sigma2*np.exp(-distance_matrix(x,y)/l))
    return(evalCov)

def indCov(sigma2):
    def evalCov(x,y):
        if np.linalg.norm(x-y)==0:
            return(sigma2)
        else:
            return(0)
    return(evalCov)     

def rMultNorm(n,mu,Sigma):
    Z = random.normal(size=(n,1))
    L = np.linalg.cholesky(Sigma)
    return(np.matmul(L,Z)+mu)
    
    
def zeroMean(x):
    return(0) 
    
    
    
