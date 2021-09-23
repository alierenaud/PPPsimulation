# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:51:32 2021

@author: alier
"""

import numpy as np
from numpy import random
from scipy.stats import beta
from rGP import GP
#from rGP import gaussianCov
from rGP import expCov
from rGP import rMultNorm
#from rGP import indCov
from rppp import PPP
from rppp import mtPPP
from scipy.stats import gamma
import matplotlib.pyplot as plt
import scipy as sp
from dmatrix import bdmatrix
from dmatrix import dsymatrix

from scipy.stats import matrix_normal
from scipy.stats import wishart

from matplotlib.backends.backend_pdf import PdfPages

# Source: (Adams, 2009) Tractable Nonparametric Bayesian Inference in Poisson Processes
# with Gaussian Process Intensities



### sampling number of thinned events (3.3) ###

### 
# Takes the location of all points and returns a new one drawn uniformly 
# along with the its conditionnal GP value
###

def insProp(lam, thisGP, locations, values, Rmat, Tmat, mu):
    
    newLoc =  random.uniform(size=(2))
    
    
    ## propose new value from MGP(.|totVal)
    
    s_11 = thisGP.cov([newLoc],[newLoc])
    S_21 = thisGP.cov(locations.totLoc(),[newLoc])
    
    S_12S_22m1 = np.dot(np.transpose(S_21),Rmat.inver)
    
    mu_row = np.dot(S_12S_22m1,values.totLoc()-mu)+mu
    spatSig = s_11 - np.dot(S_12S_22m1,S_21)
    
    A = np.linalg.cholesky(Tmat)
    
    K = Tmat.shape[0]
    Am = sp.linalg.solve_triangular(A,np.identity(K),lower=True)
    
    newVal = np.sqrt(spatSig)*np.random.normal(size=(1,K))@Am+mu_row
    
    acc_ins = (1-b(locations.nThin+1))/b(locations.nThin)*lam/(locations.nThin+1)/(1+np.sum(np.exp(newVal)))
        
    U = random.uniform(size=1)
        
    if U < acc_ins:
        
        locations.birth(newLoc)
        values.birth(newVal)
        

        
        Rmat.concat(S_21,s_11)
    
    
    




# ### TESTER: insProp
def zeroMean(x):
    return(0) 

# newGP = GP(zeroMean,gaussianCov(1,1))

# lam=25
# pointpo = PPP.randomHomog(lam)

# resGP = newGP.rGP(pointpo.loc)

# Sigma = newGP.covMatrix(pointpo.loc)

# insProp(newGP, pointpo.loc, resGP, Sigma)
# ###

### 
# Takes the location of thinned points and removes 1 uniformly
###

def delProp(lam, locations, values, Sigma):
    

    
    delInd = random.choice(np.array(range(0,locations.nThin)))
    
    oldVal = values.getThinLoc(delInd)
    
    acc_del = b(locations.nThin-1)/(1-b(locations.nThin))*locations.nThin/lam*(1+np.sum(np.exp(oldVal)))
                                                  
    U = random.uniform(size=1)
        
    if U < acc_del:
        locations.death(delInd)
        values.death(delInd)
        

        
        Sigma.delete(delInd)
    
    

    


# ### TESTER: delProp
# newGP = GP(zeroMean,gaussianCov(1,1))

# lam=25
# pointpo = PPP.randomHomog(lam)

# resGP = newGP.rGP(pointpo.loc)

# delProp(pointpo.loc, resGP)
# ###

###
# take number of thin events and return the prob of insertion
###

def b(nthin):
    if nthin==0:
        return(1)
    else:
        return(0.5)

# ### TESTER: b
# b(0)
# b(random.choice(np.array(range(0,5))))


###
# take number of thin events and return the prob of ins\del
###

def alpha(nthin):
    if nthin==0:
        return(1)
    else:
        return(0.70)

# ### TESTER: b
# alpha(0)
# alpha(random.choice(np.array(range(0,5))))

###
# inversion of inside block of matrix
###

# def woodDelInv(Sigma,Sigma_inv,i):
    
#     n = Sigma.shape[0]

#     V = np.concatenate(([Sigma[i,:]],[np.zeros(n)]))
#     V[:,i] = [0,1] 

#     U = np.transpose(V)[:,::-1]
    

#     B_inv = Sigma_inv + Sigma_inv@U@np.linalg.inv(np.identity(2)-V@Sigma_inv@U)@V@Sigma_inv

#     B_inv_del = np.delete(np.delete(B_inv,i,0),i,1)


    
#     return(B_inv_del)


###
# inserts or deletes a thinned event along with the GP value
###

def nthinSampler(lam, thisGP, locations, values, Rmat, Tmat, mu):
    

    
    B = random.binomial(1,b(locations.nThin),1)
    
    if B:

        
        insProp(lam, thisGP, locations, values, Rmat, Tmat, mu)
        
        # acc_ins = (1-b(nthin+1))/b(nthin)*lam/(nthin+1)/(1+np.exp(newVal))
        
        # U = random.uniform(size=1)
        
        # if U < acc_ins:
        #     locThin = np.concatenate((newLoc, locThin))
        #     valThin = np.concatenate((newVal, valThin))
        #     Sigma=newSigma
        #     Sigma_inv=newSigma_inv
    else:
        delProp(lam, locations, values, Rmat)
        
        # acc_del = b(nthin-1)/(1-b(nthin))*nthin/lam*(1+np.exp(oldVal))
                                                  
        # U = random.uniform(size=1)
        
        # if U < acc_del:
        #     locThin = newThinLocs
        #     valThin = newThinVal
        #     Sigma_inv = woodDelInv(Sigma,Sigma_inv,delInd)
        #     Sigma = np.delete(np.delete(Sigma, delInd, 0), delInd, 1)
            
            


    
# ### TESTER: nthinSampler
# newGP = GP(zeroMean,gaussianCov(1,1))

# lam=25
# pointpo = PPP.randomHomog(lam)

# resGP = newGP.rGP(pointpo.loc) 
# Sigma = newGP.covMatrix(pointpo.loc)


# locThin = np.empty((0,2))
# valThin = np.empty((0,1))

    
# locThin, valThin, Sigma = nthinSampler(25, newGP, locThin, valThin, pointpo.loc, resGP,Sigma)    
# locThin, valThin 


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

# ### TESTER: jitterBeta
# jitterBeta(0.5,10)
# jitterBeta(0.5,3)
# jitterBeta(0.9,10)


###
# kernel of the beta(kappa) jitter from (2D) xOld to xNew
###

def kernelBeta(xNew,xOld,kappa):
    phi0 = kappa/min(xOld[0],1-xOld[0])
    phi1 = kappa/min(xOld[1],1-xOld[1])
    d0 = beta.pdf(xNew[0], xOld[0]*phi0, (1-xOld[0])*phi0)
    d1 = beta.pdf(xNew[1], xOld[1]*phi1, (1-xOld[1])*phi1)
    return(d0*d1)

# ### TESTER: kernelBeta
# xOld = np.array([[0.5,0.5]])
# xNew = np.array([[0.6,0.6]])
# kappa = 10

# kernelBeta(xNew, xOld, kappa)


# xOld = np.array([[0.5,0.5]])
# xNew = np.array([[0.7,0.7]])
# kappa = 10

# kernelBeta(xNew, xOld, kappa)


# xOld = np.array([[0.5,0.5]])
# xNew = np.array([[0.4,0.4]])
# kappa = 10

# kernelBeta(xNew, xOld, kappa)


# xOld = np.array([[0.7,0.7]])
# xNew = np.array([[0.8,0.8]])
# kappa = 10

# kernelBeta(xNew, xOld, kappa)

# xOld = np.array([[0.7,0.7]])
# xNew = np.array([[0.9,0.9]])
# kappa = 10

# kernelBeta(xNew, xOld, kappa)


###
# Jitters one of the point process location and resamples the GP at said location
###

def locationMove(kappa, thisGP, locations, values, Rmat, Tmat, mu):

    
    moveInd = random.choice(np.array(range(0,locations.nThin))) ## choose random point to move
    
    ## propose new point
    moveLoc = locations.getThinLoc(moveInd)
    newLoc = np.array([jitterBeta(moveLoc[0],kappa),jitterBeta(moveLoc[1],kappa)])


    ## propose new value from MGP(.|totVal)
    
    s_11 = thisGP.cov([newLoc],[newLoc])
    S_21 = thisGP.cov(locations.totLoc(),[newLoc])
    
    S_12S_22m1 = np.dot(np.transpose(S_21),Rmat.inver)
    
    mu_row = np.dot(S_12S_22m1,values.totLoc()-mu)+mu
    spatSig = s_11 - np.dot(S_12S_22m1,S_21)
    
    A = np.linalg.cholesky(Tmat)
    
    K = Tmat.shape[0]
    Am = sp.linalg.solve_triangular(A,np.identity(K),lower=True)
    
    newVal = np.sqrt(spatSig)*np.random.normal(size=(1,K))@Am+mu_row
    
    
    ## accept-reject
    acc_loc = kernelBeta(moveLoc, newLoc, kappa)/(1+np.sum(np.exp(newVal)))*(1+np.sum(np.exp(values.getThinLoc(moveInd))))/kernelBeta(newLoc, moveLoc, kappa) 
        
    U = random.uniform(size=1)
        
    if U < acc_loc:
        
        locations.move(moveInd,newLoc)
        values.move(moveInd,newVal)
        
        S_21[locations.nObs+moveInd,:] = s_11
        
        Rmat.change(S_21,moveInd)


# ### TESTER locationMove
# kappa=10
# newGP = GP(zeroMean,gaussianCov(1,1))

# lam=4
# pointpo = PPP.randomHomog(lam)

# locThin = pointpo.loc
# valThin = newGP.rGP(pointpo.loc) 

# pointpo = PPP.randomHomog(lam)

# locObs = pointpo.loc
# valObs = newGP.rCondGP(locObs, locThin, valThin) 
# Sigma = newGP.covMatrix(np.concatenate((locThin,locObs)))
# print(locThin, valThin) 
    
# locThinNew, valThinNew, Sigma = locationMove(kappa, newGP, locThin, valThin, locObs, valObs, Sigma)    
# print(locThinNew, valThinNew) 
 


###
# Combines a mixture of birth-death-move type of samplers
###

def birthDeathMove(lam, kappa, thisGP, locations, values, Rmat, Tmat, mu):
    
    
    A = random.binomial(1,alpha(locations.nThin),1)
    
    if A:
        nthinSampler(lam, thisGP, locations, values, Rmat, Tmat, mu)
    else:
        locationMove(kappa, thisGP, locations, values, Rmat, Tmat, mu)
    


# ### TESTER birthDeathMove

# kappa=10
# newGP = GP(zeroMean,gaussianCov(1,0.1))

# lam=4
# pointpo = PPP.randomHomog(lam)

# locThin = pointpo.loc
# valThin = newGP.rGP(pointpo.loc) 

# pointpo = PPP.randomHomog(lam)

# locObs = pointpo.loc
# valObs = newGP.rCondGP(locObs, locThin, valThin) 
# Sigma = newGP.covMatrix(np.concatenate((locThin,locObs)))
# print(locThin, valThin) 
    
# locThinNew, valThinNew, Sigma = birthDeathMove(lam, kappa, newGP, locThin, valThin, locObs, valObs, Sigma)    
# print(locThinNew, valThinNew)

###
# loops through thinned points and proposes new locations and GP values
###

#def locationSampler(kappa, thisGP, locThin, valThin, locObs, valObs):
#    
#    i=0
#    for thisLoc in locThin:
#        thisLoc = np.array([thisLoc])
#        newLoc = np.array([[jitterBeta(thisLoc[:,0],kappa),jitterBeta(thisLoc[:,1],kappa)]])
#        
#        locTot = np.concatenate((locThin, locObs))
#        valTot = np.concatenate((valThin, valObs))
#        newVal = thisGP.rCondGP(newLoc, locTot, valTot)
#        
#        acc_loc = kernelBeta(thisLoc, newLoc, kappa)/(1+np.exp(newVal))*(1+np.exp(valThin[i]))/kernelBeta(newLoc, thisLoc, kappa) 
#        
#        U = random.uniform(size=1)
#        
#        if U < acc_loc:
#            locThin[i] = newLoc
#            valThin[i] = newVal
#        
#        i+=1
#    return(locThin, valThin)

# ### TESTER locationSampler
# kappa=10
# newGP = GP(zeroMean,gaussianCov(1,1))

# lam=10
# pointpo = PPP.randomHomog(lam)

# locThin = pointpo.loc
# valThin = newGP.rGP(pointpo.loc) 

# pointpo = PPP.randomHomog(lam)

# locObs = pointpo.loc
# valObs = newGP.rCondGP(locObs, locThin, valThin) 

    
# locThin, valThin = locationSampler(kappa, newGP, locThin, valThin, locObs, valObs)    
# locThin, valThin 

###
# expit
###

def expit(x):
    return(np.exp(x)/(1+np.exp(x)))

# ### TESTER: expit
# arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
# arr

# expit(arr)


###
# returns lower triangular matrix with half diagonal
###

def PHI(X):
    i=0
    for x in X:
        x[i] = x[i]/2
        x[(i+1):] = 0
        i += 1
    return X



# ### TESTER: PHI
# PHI(np.array([[1.0,2,3],[4,5,6],[7,8,9]]))



###
# potential energy in spatially whitened space (AA^T=Rmat) (GP values + range parameter)
###

def U(whiteVal, A, nObs, Tmat, mu, typeMatrix, rho, a, b):
    
    
    AH = A@whiteVal+mu
    
    
    K = whiteVal.shape[1]
    One_K = np.ones(shape=(K,1))
    
    
    
    return(1/2*np.trace(Tmat@np.transpose(whiteVal)@whiteVal)
           -np.sum(AH[:nObs]*np.transpose(typeMatrix))
           +np.sum(np.log(1+np.exp(AH)@One_K))
           -(a-1)*np.log(rho) + b*rho)


# ### TESTER: U

# rho=2

# thisGP = GP(zeroMean,expCov(1,rho))

# pp1 = PPP.randomHomog(5)
# pp2 = PPP.randomHomog(5)
# pp3 = PPP.randomHomog(5)


# pps = np.array([pp1,pp2,pp3])

# mtpp = mtPPP(pps)

# ppThin = PPP.randomHomog(5)

# Rmat = thisGP.covMatrix(np.concatenate((mtpp.locs,ppThin.loc)))

# K=mtpp.K
# Tmat = np.identity(K)

# nObs = mtpp.nObs
# typeMatrix = mtpp.typeMatrix

# val = matrix_normal.rvs(rowcov=Rmat,colcov=np.linalg.inv(Tmat))

# A = np.linalg.cholesky(Rmat)
# Ainv = sp.linalg.solve_triangular(A,np.identity(A.shape[0]),lower=True)

# whiteVal = Ainv@val

# a1=2
# b1=2

# U(whiteVal, A, nObs, Tmat, typeMatrix, rho, a1, b1)


###
# potential energy in whitened space (AA^T=Sigma) 
###

# def U(whiteVal, A, nObs, typeMatrix):
    
#     return(np.sum(np.log(1+np.exp(np.dot(-A[:nObs,:],whiteVal)))) + 
#            np.sum(np.log(1+np.exp(np.dot(A[nObs:,:],whiteVal)))) +
#            1/2*np.sum(whiteVal**2))


###
# potential energy in whitened space (AA^T=Sigma) +  rho
###

# def Urange(whiteVal, A, nObs, rho, a, b):
    
#     return(U(whiteVal, A, nObs) - (a-1)*np.log(rho) +b*rho)


    
# ### TESTER: U
# newGP = GP(zeroMean,gaussianCov(1,1))

# lam=10
# pointpo = PPP.randomHomog(lam) 

# locTot = pointpo.loc
# Sigma = newGP.covMatrix(locTot)
# valTot = newGP.rGP(pointpo.loc)

# nthin = 3

# A = np.linalg.cholesky(Sigma)
# whiteVal = np.linalg.inv(A)@valTot

# U(whiteVal,A,nthin)


###
# returns (1) derivative of potential energy wrt spatially whitened function values H (2) wrt to range parameter rho
###

def Uprime(whiteVal, A, Ainv, nObs, Tmat, mu, typeMatrix, Rmat, rho, a, b):
    
    expAH = np.exp(A@whiteVal+mu)
    
    K = whiteVal.shape[1]
    One_K = np.ones(shape=(K,1))
    
    OnepExpAHOne_Km1 = 1/(1+expAH@One_K)
    
    ntot = A.shape[0]
    name = np.zeros(shape=(ntot,ntot))
    name[:nObs] = np.transpose(whiteVal@typeMatrix)
    
    return(whiteVal@Tmat + np.transpose(A*OnepExpAHOne_Km1)@expAH-np.transpose(typeMatrix@A[:nObs]),
           np.sum((expAH@np.transpose(whiteVal)*OnepExpAHOne_Km1 - name)*(A@PHI(Ainv@(Rmat*(np.log(Rmat)/rho))@np.transpose(Ainv))))-(a-1)/rho+b)
           

### TESTER: Uprime

# rho=2

# thisGP = GP(zeroMean,expCov(1,rho))

# pp1 = PPP.randomHomog(5)
# pp2 = PPP.randomHomog(5)
# pp3 = PPP.randomHomog(5)


# pps = np.array([pp1,pp2,pp3])

# mtpp = mtPPP(pps)

# ppThin = PPP.randomHomog(5)

# Rmat = thisGP.covMatrix(np.concatenate((mtpp.locs,ppThin.loc)))

# K=mtpp.K
# Tmat = np.identity(K)

# nObs = mtpp.nObs
# typeMatrix = mtpp.typeMatrix

# val = matrix_normal.rvs(rowcov=Rmat,colcov=np.linalg.inv(Tmat))

# A = np.linalg.cholesky(Rmat)
# Ainv = sp.linalg.solve_triangular(A,np.identity(A.shape[0]),lower=True)

# whiteVal = Ainv@val

# a1=2
# b1=2

# Uprime(whiteVal, A, Ainv, nObs, Tmat, typeMatrix, Rmat, rho, a1, b1)



###
# derivative of potential energy in whitened space (AA^T=Sigma)
###

# def U_prime(whiteVal, A, nObs):

#     return(np.transpose(np.dot(np.transpose(expit(np.dot(A[nObs:,:],whiteVal))),A[nObs:,:]))
#            -np.transpose(np.dot(np.transpose(expit(np.dot(-A[:nObs,:],whiteVal))),A[:nObs,:]))
#            +whiteVal)



###
# derivative of potential energy in whitened space (AA^T=Sigma) + rho
###

# def Urange_prime(whiteVal, A, Ainv, Sigma, nObs, rho, tau, a ,b):
    
#     ntot = whiteVal.shape[0]
#     vec = np.zeros(shape=(ntot+1,1))
    
    
#     vec[0:ntot,:] = U_prime(whiteVal,A,nObs)
    
#     H = expit(-A@whiteVal)
#     H[nObs:] = H[nObs:]-1
    
#     vec[ntot,:] = -(a-1)/rho + b - np.sum(H@np.transpose(whiteVal)*A@PHI(Ainv@(Sigma*np.log(tau*Sigma)/rho)@np.transpose(Ainv)))

#     return(vec)



# ### TESTER: U_prime
# newGP = GP(zeroMean,gaussianCov(1,1))

# lam=10
# pointpo = PPP.randomHomog(lam) 

# locTot = pointpo.loc
# Sigma = newGP.covMatrix(locTot)
# valTot = newGP.rGP(pointpo.loc)

# nthin = 3

# A = np.linalg.cholesky(Sigma)
# whiteVal = np.linalg.inv(A)@valTot

# U_prime(whiteVal,A,nthin)





###
# sampling the GP values at the thinned and observed events
###

# def functionSampler(delta,L,values,Sigma):
    
#     A = np.linalg.cholesky(Sigma.sliceMatrix())
    
#     nObs = values.nObs
#     ntot = values.nThin + nObs
#     whiteVal = np.dot(sp.linalg.solve_triangular(A,np.identity(ntot),lower=True),values.totLoc())
    
#     kinVal = random.normal(size=(ntot,1))
    
#     kinVal_prime = kinVal - delta/2*U_prime(whiteVal, A, nObs)
#     whiteVal_prime = whiteVal + delta*kinVal_prime
    
#     l=0
#     while(l<L):
#         kinVal_prime = kinVal_prime - delta*U_prime(whiteVal_prime,A,nObs)
#         whiteVal_prime = whiteVal_prime + delta*kinVal_prime
        
#         l += 1
        
#     kinVal_prime = kinVal_prime - delta/2*U_prime(whiteVal_prime,A,nObs)
    
#     a_func = np.exp(-U(whiteVal_prime,A,nObs)+U(whiteVal,A,nObs)
#                     - 1/2*np.sum(kinVal_prime**2)
#                     + 1/2*np.sum(kinVal**2))
    
#     Uf = random.uniform(size=1)
        
#     if Uf < a_func:
#         values.newVals(np.dot(A,whiteVal_prime))
        
# ### TESTER: functionSampler
# newGP = GP(zeroMean,gaussianCov(1,1))

# lam=10
# pointpo = PPP.randomHomog(lam) 

# locTot = pointpo.loc
# Sigma = newGP.covMatrix(locTot)
# valTot = newGP.rGP(pointpo.loc)

# nthin = 3
# delta = 0.1
# L = 10

# A = np.linalg.cholesky(Sigma)
# whiteVal = np.linalg.inv(A)@valTot

# valTot
# whiteVal = functionSampler(delta,L,whiteVal,A,nthin)

# A@whiteVal

### ###

# newGP = GP(zeroMean,gaussianCov(1,1))

# lam=10
# pointpo = PPP.randomHomog(lam) 

# locTot = np.array([[0.1,0.1],[0.2,0.2],[0.3,0.3],[0.8,0.8],[0.9,0.9]])
                    
# Sigma = newGP.covMatrix(locTot)
# valTot = newGP.rGP(locTot)

# nthin = 3
# delta = 0.1
# L = 10

# A = np.linalg.cholesky(Sigma)
# whiteVal = np.linalg.inv(A)@valTot

# valTot
# whiteVal = functionSampler(delta,L,whiteVal,A,nthin)

# A@whiteVal        
    
###
# sampling the GP values at the thinned and observed events + the range parameter rho
###

def functionRangeSampler(delta,L,values,Rmat,rho,Tmat,mu,typeMatrix,a,b,GP_mom_scale,range_mom_scale):
    
    
    

    
    R_temp = Rmat.sliceMatrix()
    A = np.linalg.cholesky(R_temp)
    
    nObs = values.nObs
    ntot = values.nThin + nObs
    
    Ainv = sp.linalg.solve_triangular(A,np.identity(ntot),lower=True)
    whiteVal = np.dot(Ainv,values.totLoc()-mu)
    
    K = Tmat.shape[0]
    H_mom_init = random.normal(size=(ntot,K))*GP_mom_scale
    rho_mom_init = random.normal()*range_mom_scale
    
    
    ### leapfrog algorithm
    Uprime_H, Uprime_rho = Uprime(whiteVal,A,Ainv,nObs,Tmat,mu,typeMatrix,R_temp,rho,a,b)
    
    H_mom = H_mom_init - delta/2*Uprime_H
    rho_mom = rho_mom_init - delta/2*Uprime_rho
    
    H_pos = whiteVal + delta*H_mom/GP_mom_scale**2
    rho_pos = rho + delta*rho_mom/range_mom_scale**2
    if rho_pos < 0:
            rho_pos *= -1
            rho_mom *= -1
    
    rho_prev = rho
    
    l=0
    while(l<L):
        rho_temp = rho_pos
        
        R_temp = R_temp**(rho_temp/rho_prev)
        A_temp = np.linalg.cholesky(R_temp)
        Ainv_temp = sp.linalg.solve_triangular(A_temp,np.identity(ntot),lower=True)
        
        Uprime_H, Uprime_rho = Uprime(H_pos,A_temp,Ainv_temp,nObs,Tmat,mu,typeMatrix,R_temp,rho_pos,a,b)
    
        H_mom = H_mom - delta/2*Uprime_H
        rho_mom = rho_mom - delta/2*Uprime_rho
    
        H_pos = H_pos + delta*H_mom/GP_mom_scale**2
        rho_pos = rho_pos + delta*rho_mom/range_mom_scale**2
        if rho_pos < 0:
            rho_pos *= -1
            rho_mom *= -1
        
        rho_prev = rho_temp
        
        l += 1
        
    R_temp = R_temp**(rho_pos/rho_prev)
    A_temp = np.linalg.cholesky(R_temp)
    Ainv_temp = sp.linalg.solve_triangular(A_temp,np.identity(ntot),lower=True)    
        
    Uprime_H, Uprime_rho = Uprime(H_pos,A_temp,Ainv_temp,nObs,Tmat,mu,typeMatrix,R_temp,rho_pos,a,b)
    
    H_mom = H_mom - delta/2*Uprime_H
    rho_mom = rho_mom - delta/2*Uprime_rho

    
    a_func = np.exp(-U(H_pos, A_temp, nObs, Tmat, mu, typeMatrix, rho_pos, a, b)
                    +U(whiteVal, A, nObs, Tmat, mu, typeMatrix, rho, a, b)
                    - 1/2/GP_mom_scale**2*np.sum(H_mom**2) - 1/2/range_mom_scale**2*rho_mom**2
                    + 1/2/GP_mom_scale**2*np.sum(H_mom_init**2) + 1/2/range_mom_scale**2*rho_mom_init**2)
    
    Uf = random.uniform(size=1)
        
    if Uf < a_func:
        values.newVals(np.dot(A_temp,H_pos)+mu)
        rho = rho_pos
        Rmat.reinit(R_temp)
    
    return rho


# ### TESTER: functionRangeSampler
# newGP = GP(zeroMean,expCov(1,1))

# lam=10
# pointpo = PPP.randomHomog(lam) 

# locTot = pointpo.loc


# ### cov matrix initialization
# nObs = 3
    
# Sigma = dsymatrix(100,newGP.covMatrix(locTot),nObs)
    
# ### GP values container initialization
    
# values = bdmatrix(100,newGP.rGP(locTot),nObs,"values")


# delta = 0.01
# L = 100

# a=5
# b1=5

# rho=1

# functionRangeSampler(delta,L,values,Sigma,rho,a,b1)

# A = np.linalg.cholesky(Sigma)
# whiteVal = np.linalg.inv(A)@valTot

# valTot
# whiteVal = functionSampler(delta,L,whiteVal,A,nthin)

# A@whiteVal

### ###

# newGP = GP(zeroMean,gaussianCov(1,1))

# lam=10
# pointpo = PPP.randomHomog(lam) 

# locTot = np.array([[0.1,0.1],[0.2,0.2],[0.3,0.3],[0.8,0.8],[0.9,0.9]])
                    
# Sigma = newGP.covMatrix(locTot)
# valTot = newGP.rGP(locTot)

# nthin = 3
# delta = 0.1
# L = 10

# A = np.linalg.cholesky(Sigma)
# whiteVal = np.linalg.inv(A)@valTot

# valTot
# whiteVal = functionSampler(delta,L,whiteVal,A,nthin)

# A@whiteVal


###
# base intensity lambda sampler
###

def intensitySampler(mu,sigma2,ntot):
    alpha=mu**2/sigma2 + ntot
    beta=mu/sigma2 + 1
    lam = gamma.rvs(alpha, scale=1/beta)
    return(lam)


# ### TESTER: intensity sampler
# mu = 100
# sigma2 = 100

# intensitySampler(mu,sigma2,500)



###
# precision tau sampler
###

# def precisionSampler(a_tau,b_tau,tau_prev,values,Sigma):
#     ntot = values.nThin + values.nObs
#     alpha=a_tau + ntot/2
#     beta=b_tau + np.transpose(values.totLoc())@Sigma.inver@values.totLoc()/2/tau_prev
#     tau = gamma.rvs(alpha, scale=1/beta)
    
#     Sigma.rescale(tau_prev/tau)
#     return(tau)

###
# type precision T sampler
###

def typePrecisionSampler(n,Vm1,values,Rmat,mu):
    ntot = values.nThin + values.nObs
    n_post=n + ntot
    V_post=Vm1 + np.transpose(values.totLoc()-mu)@Rmat.inver@(values.totLoc()-mu)
    Tmat= wishart.rvs(n_post, np.linalg.inv(V_post))
    
    # Sigma.rescale(tau_prev/tau)
    return(Tmat)

###
# type mean mu sampler values,Rmat,Ts[i],var_mu
###

def typeMeanSampler(values,Rmat,Tmat,mean_mu,var_mu):
    N = values.nThin + values.nObs
    K = Tmat.shape[0]
    Rm1IndN = Rmat.inver @ np.ones(shape=(N,1))
    Sigma_star = np.linalg.inv((np.ones(shape=(1,N)) @ Rm1IndN) *Tmat + np.identity(K)/var_mu)
    mu_star = Sigma_star @ (Tmat @ np.transpose(values.totLoc()) @ Rm1IndN + np.transpose(mean_mu)/var_mu)
        
    mu = np.linalg.cholesky(Sigma_star)@np.random.normal(size=(K,1)) + mu_star
    return(np.transpose(mu))



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


def multExpit(x):
    N = np.sum(np.exp(x))
    probs = np.array([np.exp(i)/(1+N) for i in x])
    return(np.append(probs,1-np.sum(probs)))


###
# full sampler of thinned locations, thinned values, observed values and intensity
###


def MCMCadams(size,lam_init,rho_init,T_init,thismtPP,nInsDelMov,kappa,delta,L,mu_lam,sigma2,p,a,b,n,V,mu_init,mean_mu,var_mu,diagnostics,res,thin,GP_mom_scale,range_mom_scale):
    
    
    ### independent type prior mean
    
    ### initialize GP
    thisGP = GP(zeroMean,expCov(1,rho_init))
    
    ### location container initialization
    K = thismtPP.K
    totLocInit = np.concatenate((thismtPP.locs,PPP.randomHomog(lam=int(lam_init//(K+1))).loc),0)
    nObs = thismtPP.nObs
    
    locations = bdmatrix(int(10*lam_init),totLocInit,nObs,"locations") # initial size is a bit of black magic
    
    ### cov matrix initialization
    
    Rmat = dsymatrix(int(10*lam_init),thisGP.covMatrix(totLocInit),nObs)
    
    ### GP values container initialization
    
    
    # ### try to initiate GP in logical position
    
    # mean_init = np.zeros(shape=(locations.nThin+nObs,K))
    
    # mean_init[:nObs,:] = np.transpose(np.linalg.cholesky(np.linalg.inv(T_init))@thismtPP.typeMatrix)
    
    # ####
    
    
    values = bdmatrix(int(10*lam_init),matrix_normal.rvs(rowcov=Rmat.sliceMatrix(),colcov=np.linalg.inv(T_init)) + mu_init,nObs,"values")
    
    
    
    
    
    ### parameters containers
    lams = np.empty(shape=(size))
    rhos = np.empty(shape=(size))
    
    Ts = np.empty(shape=(size,K,K))
    mus = np.empty(shape=(size,1,K))
    Nthins = np.empty(shape=(size))
    
    ### independent type prior mean
    Vm1 = np.linalg.inv(V)
    
    
    ### 
    lams[0] = lam_init
    rhos[0] = rho_init
    Ts[0] = T_init
    mus[0] = mu_init
    Nthins[0] = locations.nThin
    
    
    ### instantiate containers for diagnostics
    if diagnostics:
        danceLocs = np.empty(shape=(int(size//thin),int(10*lam_init),2))
        GPvaluesAtObs = np.empty(shape=(int(size//thin),nObs,K))
        fieldsGrid = np.empty(shape=(int(size//thin),res**2,K+1))
        
        danceLocs[0,:int(nObs+Nthins[0])] = locations.totLoc()
        GPvaluesAtObs[0] = values.obsLoc()
        
        gridLoc = makeGrid([0,1], [0,1], res)
        
        s_11 = thisGP.cov(gridLoc,gridLoc)
        S_21 = thisGP.cov(locations.totLoc(),gridLoc)
        
        S_12S_22m1 = np.dot(np.transpose(S_21),Rmat.inver)
    
        muGrid = np.dot(S_12S_22m1,values.totLoc()-mus[0]) + mus[0]
        spatSig = s_11 - np.dot(S_12S_22m1,S_21)
        
        A = np.linalg.cholesky(Ts[0])
    

        Am = sp.linalg.solve_triangular(A,np.identity(K),lower=True)
        
        newVal = np.linalg.cholesky(spatSig)@np.random.normal(size=(res**2,K))@Am+muGrid
    
    
        fieldsGrid[0] = lams[0]*np.array([multExpit(val) for val in newVal])
        
    
    
    i=1
    diagnb = 1
    while i < size:
        
        j=0
        while j < nInsDelMov:
            birthDeathMove(lams[i-1],kappa,thisGP,locations,values,Rmat,Ts[i-1],mus[i-1])
            j+=1
        

        Nthins[i] = locations.nThin        

        # # locTot_prime = np.concatenate((locThin_prime,thisPPP.loc))
        # valTot_prime = np.concatenate((valThin_prime,obsVal[i-1]))
        
        # nthin = locThin_prime.shape[0]
        
        # # Sigma = thisGP.covMatrix(locTot_prime)
        # A = np.linalg.cholesky(Sigmas[i])
        # ntot = A.shape[0]
        
        # whiteVal_prime = sp.linalg.solve_triangular(A,np.identity(ntot),lower=True)@valTot_prime
        
        # functionSampler(delta,L,values,Sigma)
        
        rhos[i] = functionRangeSampler(delta,L,values,Rmat,rhos[i-1],Ts[i-1],mus[i-1],thismtPP.typeMatrix,a,b,GP_mom_scale,range_mom_scale)
        Ts[i] = typePrecisionSampler(n,Vm1,values,Rmat,mus[i-1])
        mus[i] = typeMeanSampler(values,Rmat,Ts[i],mean_mu,var_mu)
        thisGP = GP(zeroMean,expCov(1,rhos[i]))
        
        
        
        # valTot_prime = A @ whiteVal_prime
        
        # thinLoc[i] = locThin_prime
        # thinVal[i] = valTot_prime[:nthin,:]
        # obsVal[i] = valTot_prime[nthin:,:]
        
        # ntot = valTot_prime.shape[0]
        
        lams[i] = intensitySampler(mu_lam,sigma2,values.nThin + values.nObs)
        
        if diagnostics and i%thin == 0:
            
        
            danceLocs[diagnb,:int(nObs+Nthins[i])] = locations.totLoc()
            GPvaluesAtObs[diagnb] = values.obsLoc()
        
        
            s_11 = thisGP.cov(gridLoc,gridLoc)
            S_21 = thisGP.cov(locations.totLoc(),gridLoc)
        
            S_12S_22m1 = np.dot(np.transpose(S_21),Rmat.inver)
    
            muGrid = np.dot(S_12S_22m1,values.totLoc()-mus[i]) + mus[i]
            spatSig = s_11 - np.dot(S_12S_22m1,S_21)
        
            A = np.linalg.cholesky(Ts[i])
    

            Am = sp.linalg.solve_triangular(A,np.identity(K),lower=True)
        
            newVal = np.linalg.cholesky(spatSig)@np.random.normal(size=(res**2,K))@Am+muGrid
    
    
            fieldsGrid[diagnb] = lams[i]*np.array([multExpit(val) for val in newVal])
            
            
            diagnb += 1
        
        
        if p:
            ### next sample
            locations.nextSamp()
            values.nextSamp()
        
        
        print(i)
        i+=1
    
    
    if diagnostics:
        
        
        
        ### dancing locations plot
        mpdf = PdfPages('0thinLocs.pdf')


        diagnb=0
        while(diagnb < int(size//thin)):
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
        
        
            plt.plot(danceLocs[diagnb,nObs:int(nObs+Nthins[diagnb*thin]),0],danceLocs[diagnb,nObs:int(nObs+Nthins[diagnb*thin]),1], 'o', c=(0.75, 0.75, 0.75))

            for pp in thismtPP.pps:
                    plt.plot(pp.loc[:,0],pp.loc[:,1], 'o')
            plt.xlim(0,1)
            plt.ylim(0,1)
        
            # plt.show()
            mpdf.savefig(bbox_inches='tight') 
            plt.close(fig)
            
            
            
            diagnb+=1
        
        mpdf.close()
            
        
        ### GP traces at observerd locations    
        
        fig, axs = plt.subplots(nObs,figsize=(10,1.5*nObs))    
    
        obsNB=0
        colNB=0
        while(obsNB < nObs):
            colNB=0
            while(colNB<K):
                
                if thismtPP.typeMatrix[colNB,obsNB] == 1:
                
                    axs[obsNB].plot(GPvaluesAtObs[:,obsNB,colNB],linewidth=2)
                    
                else:
                    axs[obsNB].plot(GPvaluesAtObs[:,obsNB,colNB],linestyle="dashed")
                
                
        
                colNB+=1
            
            obsNB+=1   
        
        # plt.show()
        fig.savefig("0GPtraces.pdf", bbox_inches='tight')
        plt.close(fig)
        
        
        ### mean intensities
        
        meanFields = np.mean(fieldsGrid, axis=0, dtype=np.float32)

        maxi = np.max(meanFields)
        mini = np.min(meanFields)
        
        
        
        
        k=0
        while k<K+1:
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            
            plt.xlim(0,1)
            plt.ylim(0,1)
            
            imGP = np.transpose(meanFields[:,k].reshape(res,res))
    
            x = np.linspace(0,1, res+1) 
            y = np.linspace(0,1, res+1) 
            X, Y = np.meshgrid(x,y) 
                
            # fig = plt.figure()
            # axs[k] = fig.add_subplot(111)
            ax.set_aspect('equal')
                
            ff = ax.pcolormesh(X,Y,imGP, cmap='gray', vmin=mini, vmax=maxi)
            
            fig.colorbar(ff)   
            
            for pp in thismtPP.pps:
                ax.plot(pp.loc[:,0],pp.loc[:,1], 'o', c="tab:orange")
               
             
            
            # plt.scatter(pointpo.loc[:,0],pointpo.loc[:,1], color= "black", s=1)
                
            # plt.show()
            
            fig.savefig("0IntFields"+str(k)+".pdf", bbox_inches='tight')
            plt.close(fig)

            
            k+=1
        
        

        
        
        # fig.savefig("0IntFields.pdf", bbox_inches='tight')
        # plt.close(fig)
    
    
    return(lams, rhos, Ts, mus, Nthins)

# ### TESTER: MCMCadams



# # def fct(x):
# #      return(np.exp((-x[:,0]**2-x[:,1]**2)/0.3)*0.8+0.10)
# def fct(x):
#     return(np.exp(-(0.25-np.sqrt((x[:,0]-0.5)**2+(x[:,1]-0.5)**2))**2/0.003)*0.8+0.10)
# # def fct(x):
# #     return(np.exp(-np.minimum((x[:,0]-0.5)**2,(x[:,1]-0.5)**2)/0.007)*0.8+0.10)

# lam_sim=800

# pointpo = PPP.randomNonHomog(lam_sim,fct)
# pointpo.plot()


# newGP = GP(zeroMean,gaussianCov(2,0.5))

# niter=1000

# import time

# t0 = time.time()
# thinLoc,thinVal,obsVal,lams = MCMCadams(niter,900,newGP,pointpo,80,10,0.1,10,1000,1000)
# t1 = time.time()

# total1 = t1-t0

# # i=0
# # while(i < niter):


    

    
# #     fig = plt.figure()
# #     ax = fig.add_subplot(111)
# #     ax.set_aspect('equal')
    
# #     plt.scatter(pointpo.loc[:,0],pointpo.loc[:,1])
# #     plt.scatter(thinLoc[i][:,0],thinLoc[i][:,1])
    

# #     plt.xlim(0,1)
# #     plt.ylim(0,1)

# #     plt.show()
    
    
# #     fig.savefig("Scatter"+str(i)+".pdf", bbox_inches='tight')
# #     i+=1

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


# res = 50
# gridLoc = makeGrid([0,1], [0,1], res)



# resGP = np.empty(shape=niter,dtype=np.ndarray)
# i=0
# t0 = time.time()
# while(i < niter):
#     resGP[i] = lams[i]*expit(newGP.rCondGP(gridLoc,np.concatenate((thinLoc[i],pointpo.loc)),
#               np.concatenate((thinVal[i],obsVal[i]))))
#     print(i)
#     i+=1
# t1 = time.time()

# total2 = t1-t0

# meanGP = np.mean(resGP)


# #### to make plot ####

# imGP = np.transpose(meanGP.reshape(res,res))

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
# # fig.savefig("meanInt.pdf", bbox_inches='tight')


# #### plot of actual intensity ####

# realInt = lam_sim*fct(gridLoc)

# #### to make plot ####

# imGP = np.transpose(realInt.reshape(res,res))

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
# # fig.savefig("trueInt.pdf", bbox_inches='tight')




# ### every iteration ###

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
    
#     #plt.show()
#     fig.savefig("Int"+str(i)+".pdf", bbox_inches='tight')
#     i+=1
    
    
# i=0
# while(i < niter):


    

    
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_aspect('equal')
    
#     plt.scatter(pointpo.loc[:,0],pointpo.loc[:,1])
#     plt.scatter(thinLoc[i][:,0],thinLoc[i][:,1])
    

#     plt.xlim(0,1)
#     plt.ylim(0,1)

#     #plt.show()
    
    
#     fig.savefig("Scatter"+str(i)+".pdf", bbox_inches='tight')
#     i+=1
    
    
# ### Example with SGCP simulated data

# lam=400
# tau=1
# rho=1

# res = 50
# gridLoc = makeGrid([0,1], [0,1], res)


# locs = PPP.randomHomog(lam).loc

# newGP = GP(zeroMean,gaussianCov(tau,rho))
# GPvals = newGP.rGP(np.concatenate((locs,gridLoc)))


# gridInt = lam*expit(GPvals[locs.shape[0]:,:])


# locProb  = expit(GPvals[:locs.shape[0],:])
# index = np.array(np.greater(locProb,random.uniform(size=locProb.shape)))
# locObs = locs[np.squeeze(index)]
# locThin = locs[np.logical_not(np.squeeze(index))]

# pointpo = PPP.randomHomog(lam)
# pointpo.loc = locObs

# niter=1000

# import time

# t0 = time.time()
# thinLoc,thinVal,obsVal,lams = MCMCadams(niter,lam,newGP,pointpo,50,10,0.1,10,lam,100)
# t1 = time.time()

# total1 = t1-t0


# ### Inference at grid locations

# resGP = np.empty(shape=niter,dtype=np.ndarray)
# i=0
# t0 = time.time()
# while(i < niter):
#     resGP[i] = lams[i]*expit(newGP.rCondGP(gridLoc,np.concatenate((thinLoc[i],pointpo.loc)),
#               np.concatenate((thinVal[i],obsVal[i]))))
#     print(i)
#     i+=1
# t1 = time.time()

# total2 = t1-t0

# meanGP = np.mean(resGP)


# #### to make plot ####

# imGP = np.transpose(meanGP.reshape(res,res))

# x = np.linspace(0,1, res+1) 
# y = np.linspace(0,1, res+1) 
# X, Y = np.meshgrid(x,y) 

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_aspect('equal')

# plt.pcolormesh(X,Y,imGP, cmap='winter')

# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.colorbar()

# plt.scatter(pointpo.loc[:,0],pointpo.loc[:,1], color= "black", s=1)

# plt.show()


# ### plot of actual intensity


# ### int + obs + thin

# imGP = np.transpose(gridInt.reshape(res,res))

# x = np.linspace(0,1, res+1) 
# y = np.linspace(0,1, res+1) 
# X, Y = np.meshgrid(x,y) 

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_aspect('equal')

# plt.pcolormesh(X,Y,imGP, cmap='winter')

# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.colorbar()

# plt.scatter(locObs[:,0],locObs[:,1], color= "black", s=1)
# plt.scatter(locThin[:,0],locThin[:,1], color= "white", s=1)


# plt.show()
    

# # ###
# # # full sampler of thinned locations, thinned values, observed values and intensity
# # ###


# # def MCMCadams(size,lam_init,thisGP,thisPPP,nInsDel,kappa,delta,L,mu,sigma2):
    
# #     thinLoc = np.empty(shape=size,dtype=np.ndarray)
# #     thinVal = np.empty(shape=size,dtype=np.ndarray)
# #     obsVal = np.empty(shape=size,dtype=np.ndarray)
# #     lams = np.empty(shape=size)
    
# #     # initialization
    
# #     thinLoc[0] = PPP.randomHomog(lam=lam_init).loc
# #     thinVal[0] = thisGP.rGP(thinLoc[0])
# #     obsVal[0] = thisGP.rCondGP(thisPPP.loc,thinLoc[0],thinVal[0])
# #     lams[0] = lam_init
    
# #     i=1
# #     while i < size:
# #         locThin_prime, valThin_prime = nthinSampler(lams[i-1],thisGP,
# #                                                     thinLoc[i-1],thinVal[i-1],
# #                                                     thisPPP.loc,obsVal[i-1])
# #         j=1
# #         while j < nInsDel:
# #             locThin_prime, valThin_prime = nthinSampler(lams[i-1],thisGP,
# #                                                     locThin_prime, valThin_prime,
# #                                                     thisPPP.loc,obsVal[i-1])
# #             j+=1
        
# #         # locThin_prime, valThin_prime = locationSampler(kappa,thisGP,
# #         #                                                locThin_prime, valThin_prime,
# #         #                                                thisPPP.loc,obsVal[i-1])
        
# #         locTot_prime = np.concatenate((locThin_prime,thisPPP.loc))
# #         valTot_prime = np.concatenate((valThin_prime,obsVal[i-1]))
        
# #         nthin = locThin_prime.shape[0]
        
# #         Sigma = thisGP.covMatrix(locTot_prime)
# #         A = np.linalg.cholesky(Sigma)
        
# #         whiteVal_prime = np.linalg.inv(A)@valTot_prime
        
# #         whiteVal_prime = functionSampler(delta,L,whiteVal_prime,A,nthin)
        
# #         valTot_prime = A @ whiteVal_prime
        
# #         thinLoc[i] = locThin_prime
# #         thinVal[i] = valTot_prime[:nthin,:]
# #         obsVal[i] = valTot_prime[nthin:,:]
        
# #         ntot = valTot_prime.shape[0]
        
# #         lams[i] = intensitySampler(mu,sigma2,ntot)
        
# #         i+=1
    
    
# #     return(thinLoc,thinVal,obsVal,lams)

# # ### TESTER: MCMCadams

# # def fct(x):
# #     return(np.exp((-x[:,0]**2-x[:,1]**2)/0.3))

# # lam_sim=500

# # pointpo = PPP.randomNonHomog(lam_sim,fct)
# # pointpo.plot()


# # newGP = GP(zeroMean,gaussianCov(1,1))

# # niter=10

# # import time

# # t0 = time.time()
# # thinLoc,thinVal,obsVal,lams = MCMCadams(niter,300,newGP,pointpo,10,10,0.1,10,300,10000)
# # t1 = time.time()

# # total1 = t1-t0


# # def makeGrid(xlim,ylim,res):
# #     grid = np.ndarray((res**2,2))
# #     xlo = xlim[0]
# #     xhi = xlim[1]
# #     xrange = xhi - xlo
# #     ylo = ylim[0]
# #     yhi = ylim[1]
# #     yrange = yhi - ylo
# #     xs = np.arange(xlo, xhi, step=xrange/res) + xrange/res*0.5
# #     ys = np.arange(ylo, yhi, step=yrange/res) + yrange/res*0.5
# #     i=0
# #     for x in xs:
# #         j=0
# #         for y in ys:
# #             grid[i*res+j,:] = [x,y]
# #             j+=1
# #         i+=1
# #     return(grid)


# # res = 50
# # gridLoc = makeGrid([0,1], [0,1], res)


# # t0 = time.time()
# # resGP = expit(newGP.rCondGP(gridLoc,np.concatenate((thinLoc[9],pointpo.loc)),
# #               np.concatenate((thinVal[9],obsVal[9]))))
# # t1 = time.time()

# # total2 = t1-t0



# # #### to make plot ####

# # imGP = np.transpose(resGP.reshape(res,res))

# # x = np.linspace(0,1, res+1) 
# # y = np.linspace(0,1, res+1) 
# # X, Y = np.meshgrid(x,y) 

# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # ax.set_aspect('equal')

# # plt.pcolormesh(X,Y,imGP)

# # plt.xlim(0,1)
# # plt.ylim(0,1)


# # plt.show()


# # i=0
# # while(i < niter):
# #     resGP = expit(newGP.rCondGP(gridLoc,np.concatenate((thinLoc[i],pointpo.loc)),
# #               np.concatenate((thinVal[i],obsVal[i]))))

    
# #     imGP = np.transpose(resGP.reshape(res,res))
    
# #     x = np.linspace(0,1, res+1) 
# #     y = np.linspace(0,1, res+1) 
# #     X, Y = np.meshgrid(x,y) 
    
# #     fig = plt.figure()
# #     ax = fig.add_subplot(111)
# #     ax.set_aspect('equal')
    
# #     plt.pcolormesh(X,Y,imGP)
    
# #     plt.xlim(0,1)
# #     plt.ylim(0,1)
    
    
# #     plt.show()
# #     i+=1

# # ### ###

# # def fct(x):
# #     return(np.exp(-(0.25-np.sqrt((x[:,0]-0.5)**2+(x[:,1]-0.5)**2))**2/0.003))

# # lam_sim=500

# # pointpo = PPP.randomNonHomog(lam_sim,fct)
# # pointpo.plot()


# # newGP = GP(zeroMean,gaussianCov(1,1))

# # niter=10

# # import time

# # t0 = time.time()
# # thinLoc,thinVal,obsVal,lams = MCMCadams(niter,300,newGP,pointpo,10,10,0.1,10,300,10000)
# # t1 = time.time()

# # total1 = t1-t0


# # def makeGrid(xlim,ylim,res):
# #     grid = np.ndarray((res**2,2))
# #     xlo = xlim[0]
# #     xhi = xlim[1]
# #     xrange = xhi - xlo
# #     ylo = ylim[0]
# #     yhi = ylim[1]
# #     yrange = yhi - ylo
# #     xs = np.arange(xlo, xhi, step=xrange/res) + xrange/res*0.5
# #     ys = np.arange(ylo, yhi, step=yrange/res) + yrange/res*0.5
# #     i=0
# #     for x in xs:
# #         j=0
# #         for y in ys:
# #             grid[i*res+j,:] = [x,y]
# #             j+=1
# #         i+=1
# #     return(grid)


# # res = 50
# # gridLoc = makeGrid([0,1], [0,1], res)


# # t0 = time.time()
# # resGP = expit(newGP.rCondGP(gridLoc,np.concatenate((thinLoc[9],pointpo.loc)),
# #               np.concatenate((thinVal[9],obsVal[9]))))
# # t1 = time.time()

# # total2 = t1-t0



# # #### to make plot ####

# # imGP = np.transpose(resGP.reshape(res,res))

# # x = np.linspace(0,1, res+1) 
# # y = np.linspace(0,1, res+1) 
# # X, Y = np.meshgrid(x,y) 

# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # ax.set_aspect('equal')

# # plt.pcolormesh(X,Y,imGP)

# # plt.xlim(0,1)
# # plt.ylim(0,1)


# # plt.show()


# # i=0
# # while(i < niter):
# #     resGP = expit(newGP.rCondGP(gridLoc,np.concatenate((thinLoc[i],pointpo.loc)),
# #               np.concatenate((thinVal[i],obsVal[i]))))

    
# #     imGP = np.transpose(resGP.reshape(res,res))
    
# #     x = np.linspace(0,1, res+1) 
# #     y = np.linspace(0,1, res+1) 
# #     X, Y = np.meshgrid(x,y) 
    
# #     fig = plt.figure()
# #     ax = fig.add_subplot(111)
# #     ax.set_aspect('equal')
    
# #     plt.pcolormesh(X,Y,imGP)
    
# #     plt.xlim(0,1)
# #     plt.ylim(0,1)
    
    
# #     plt.show()
# #     i+=1


