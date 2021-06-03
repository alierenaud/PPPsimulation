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
from rGP import rMultNorm
#from rGP import indCov
from rppp import PPP
from scipy.stats import gamma
import matplotlib.pyplot as plt
import scipy as sp
from dmatrix import bdmatrix
from dmatrix import dsymatrix

# Source: (Adams, 2009) Tractable Nonparametric Bayesian Inference in Poisson Processes
# with Gaussian Process Intensities



### sampling number of thinned events (3.3) ###

### 
# Takes the location of all points and returns a new one drawn uniformly 
# along with the its conditionnal GP value
###

def insProp(lam, thisGP, locations, values, Sigma):
    
    newLoc =  random.uniform(size=(2))
    
    
    ## propose new value from GP(.|totVal)
    
    s_11 = thisGP.cov([newLoc],[newLoc])
    S_21 = thisGP.cov(locations.totLoc(),[newLoc])
    
    S_12S_22m1 = np.transpose(S_21)@Sigma.inver
    
    mu = S_12S_22m1@values.totLoc()
    sig = s_11 - S_12S_22m1@S_21
    
    newVal = np.sqrt(sig)*np.random.normal()+mu
    
    acc_ins = (1-b(locations.nThin+1))/b(locations.nThin)*lam/(locations.nThin+1)/(1+np.exp(newVal))
        
    U = random.uniform(size=1)
        
    if U < acc_ins:
        
        locations.birth(newLoc)
        values.birth(newVal)
        

        
        Sigma.concat(S_21,s_11)
    
    
    




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
    
    acc_del = b(locations.nThin-1)/(1-b(locations.nThin))*locations.nThin/lam*(1+np.exp(oldVal))
                                                  
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

def woodDelInv(Sigma,Sigma_inv,i):
    
    n = Sigma.shape[0]

    V = np.concatenate(([Sigma[i,:]],[np.zeros(n)]))
    V[:,i] = [0,1] 

    U = np.transpose(V)[:,::-1]
    

    B_inv = Sigma_inv + Sigma_inv@U@np.linalg.inv(np.identity(2)-V@Sigma_inv@U)@V@Sigma_inv

    B_inv_del = np.delete(np.delete(B_inv,i,0),i,1)


    
    return(B_inv_del)


###
# inserts or deletes a thinned event along with the GP value
###

def nthinSampler(lam, thisGP, locations, values, Sigma):
    

    
    B = random.binomial(1,b(locations.nThin),1)
    
    if B:

        
        insProp(lam, thisGP, locations, values, Sigma)
        
        # acc_ins = (1-b(nthin+1))/b(nthin)*lam/(nthin+1)/(1+np.exp(newVal))
        
        # U = random.uniform(size=1)
        
        # if U < acc_ins:
        #     locThin = np.concatenate((newLoc, locThin))
        #     valThin = np.concatenate((newVal, valThin))
        #     Sigma=newSigma
        #     Sigma_inv=newSigma_inv
    else:
        delProp(lam, locations, values, Sigma)
        
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

def locationMove(kappa, thisGP, locations, values, Sigma):

    
    moveInd = random.choice(np.array(range(0,locations.nThin))) ## choose random point to move
    
    ## propose new point
    moveLoc = locations.getThinLoc(moveInd)
    newLoc = np.array([jitterBeta(moveLoc[0],kappa),jitterBeta(moveLoc[1],kappa)])


    ## propose new value from GP(.|totVal)
    
    s_11 = thisGP.cov([newLoc],[newLoc])
    S_21 = thisGP.cov(locations.totLoc(),[newLoc])
    
    S_12S_22m1 = np.transpose(S_21)@Sigma.inver
    
    mu = S_12S_22m1@values.totLoc()
    sig = s_11 - S_12S_22m1@S_21
    
    newVal = np.sqrt(sig)*np.random.normal()+mu
    
    
    ## accept-reject
    acc_loc = kernelBeta(moveLoc, newLoc, kappa)/(1+np.exp(newVal))*(1+np.exp(values.getThinLoc(moveInd)))/kernelBeta(newLoc, moveLoc, kappa) 
        
    U = random.uniform(size=1)
        
    if U < acc_loc:
        
        locations.move(moveInd,newLoc)
        values.move(moveInd,newVal)
        
        S_21[locations.nObs+moveInd,:] = s_11
        
        Sigma.change(S_21,moveInd)


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

def birthDeathMove(lam, kappa, thisGP, locations, values, Sigma):
    
    
    A = random.binomial(1,alpha(locations.nThin),1)
    
    if A:
        nthinSampler(lam, thisGP, locations, values, Sigma)
    else:
        locationMove(kappa, thisGP, locations, values, Sigma)
    


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
# potential energy in whitened space (AA^T=Sigma) Thinned events are first in Sigma
###

def U(whiteVal, A, nObs):
    
    return(np.sum(np.log(1+np.exp(-A[:nObs,:]@whiteVal))) + 
           np.sum(np.log(1+np.exp(A[nObs:,:]@whiteVal))) +
           1/2*np.sum(whiteVal**2))
    
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
# derivative of potential energy in whitened space (AA^T=Sigma) Thinned events are first in Sigma
###

def U_prime(whiteVal, A, nObs):

    return(-np.transpose(np.transpose(expit(-A[:nObs,:]@whiteVal))@A[:nObs,:]) +
           np.transpose(np.transpose(expit(A[nObs:,:]@whiteVal))@A[nObs:,:]) +
           whiteVal)

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

def functionSampler(delta,L,values,Sigma):
    
    A = np.linalg.cholesky(Sigma.sliceMatrix())
    
    nObs = values.nObs
    ntot = values.nThin + nObs
    whiteVal = sp.linalg.solve_triangular(A,np.identity(ntot),lower=True)@values.totLoc()
    
    kinVal = random.normal(size=(ntot,1))
    
    kinVal_prime = kinVal - delta/2*U_prime(whiteVal, A, nObs)
    whiteVal_prime = whiteVal + delta*kinVal_prime
    
    l=0
    while(l<L):
        kinVal_prime = kinVal_prime - delta*U_prime(whiteVal_prime,A,nObs)
        whiteVal_prime = whiteVal_prime + delta*kinVal_prime
        
        l += 1
        
    kinVal_prime = kinVal_prime - delta/2*U_prime(whiteVal_prime,A,nObs)
    
    a_func = np.exp(-U(whiteVal_prime,A,nObs)+U(whiteVal,A,nObs)
                    - 1/2*np.sum(kinVal_prime**2)
                    + 1/2*np.sum(kinVal**2))
    
    Uf = random.uniform(size=1)
        
    if Uf < a_func:
        values.newVals(A@whiteVal_prime)
    



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
# full sampler of thinned locations, thinned values, observed values and intensity
###


def MCMCadams(size,lam_init,thisGP,thisPPP,nInsDelMov,kappa,delta,L,mu,sigma2):
    
    
    
    ### location container initialization
    totLocInit = np.concatenate((thisPPP.loc,PPP.randomHomog(lam=lam_init).loc),0)
    nObs = thisPPP.loc.shape[0]
    
    locations = bdmatrix(5*lam_init + size*nInsDelMov,totLocInit,nObs,size) # initial size is a bit of black magic
    
    ### cov matrix initialization
    
    Sigma = dsymatrix(5*lam_init,thisGP.covMatrix(totLocInit),nObs)
    
    ### GP values container initialization
    
    values = bdmatrix(4*lam_init*size,rMultNorm(0,Sigma.sliceMatrix()),nObs,size)
    
    
    ### parameters containers
    lams = np.empty(shape=(size))
    
    
### initialization of lambda
    lams[0] = lam_init
    
    i=1
    while i < size:
        
        j=0
        while j < nInsDelMov:
            birthDeathMove(lams[i-1],kappa,thisGP,locations,values,Sigma)
            j+=1
        

        
        # # locTot_prime = np.concatenate((locThin_prime,thisPPP.loc))
        # valTot_prime = np.concatenate((valThin_prime,obsVal[i-1]))
        
        # nthin = locThin_prime.shape[0]
        
        # # Sigma = thisGP.covMatrix(locTot_prime)
        # A = np.linalg.cholesky(Sigmas[i])
        # ntot = A.shape[0]
        
        # whiteVal_prime = sp.linalg.solve_triangular(A,np.identity(ntot),lower=True)@valTot_prime
        
        functionSampler(delta,L,values,Sigma)
        
        
        
        
        
        
        # valTot_prime = A @ whiteVal_prime
        
        # thinLoc[i] = locThin_prime
        # thinVal[i] = valTot_prime[:nthin,:]
        # obsVal[i] = valTot_prime[nthin:,:]
        
        # ntot = valTot_prime.shape[0]
        
        lams[i] = intensitySampler(mu,sigma2,values.nThin + values.nObs)
        
        
        ### next sample
        locations.nextSamp()
        values.nextSamp()
        
        
        print(i)
        i+=1
    
    
    return(locations, values, lams)

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


