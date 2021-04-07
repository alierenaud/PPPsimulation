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




# Source: (Adams, 2009) Tractable Nonparametric Bayesian Inference in Poisson Processes
# with Gaussian Process Intensities



### sampling number of thinned events (3.3) ###

### 
# Takes the location of all points and returns a new one drawn uniformly 
# along with the its conditionnal GP value
###

def insProp(thisGP, locObs, valObs, Sigma):
    
    newLoc =  random.uniform(size=(1, 2))
    
    valNewLoc, newSigma = thisGP.rCondGP1DSigma(newLoc, locObs, valObs, Sigma)
    
    return(newLoc, valNewLoc, newSigma)



### TESTER: insProp
def zeroMean(x):
    return(0) 

newGP = GP(zeroMean,gaussianCov(1,1))

lam=25
pointpo = PPP.randomHomog(lam)

resGP = newGP.rGP(pointpo.loc)

Sigma = newGP.covMatrix(pointpo.loc)

insProp(newGP, pointpo.loc, resGP, Sigma)
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
    
    return(oldVal,newLocThin,newValThin, delInd)

### TESTER: delProp
newGP = GP(zeroMean,gaussianCov(1,1))

lam=25
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
# take number of thin events and return the prob of ins\del
###

def alpha(nthin):
    if nthin==0:
        return(1)
    else:
        return(0.70)

### TESTER: b
alpha(0)
alpha(random.choice(np.array(range(0,5))))

###
# inserts or deletes a thinned event along with the GP value
###

def nthinSampler(lam, thisGP, locThin, valThin, locObs, valObs, Sigma):
    
    nthin = locThin.shape[0]
    
    B = random.binomial(1,b(nthin),1)
    
    if B:
        locTot = np.concatenate((locThin, locObs))
        valTot = np.concatenate((valThin, valObs))
        
        newLoc, newVal, newSigma = insProp(thisGP, locTot, valTot, Sigma)
        
        acc_ins = (1-b(nthin+1))/b(nthin)*lam/(nthin+1)/(1+np.exp(newVal))
        
        U = random.uniform(size=1)
        
        if U < acc_ins:
            locThin = np.concatenate((newLoc, locThin))
            valThin = np.concatenate((newVal, valThin))
            Sigma=newSigma
    else:
        oldVal, newThinLocs, newThinVal, delInd = delProp(locThin, valThin)
        
        acc_del = b(nthin-1)/(1-b(nthin))*nthin/lam*(1+np.exp(oldVal))
                                                  
        U = random.uniform(size=1)
        
        if U < acc_del:
            locThin = newThinLocs
            valThin = newThinVal
            Sigma = np.delete(np.delete(Sigma, delInd, 0), delInd, 1)
            

    return(locThin, valThin, Sigma)
    
### TESTER: nthinSampler
newGP = GP(zeroMean,gaussianCov(1,1))

lam=25
pointpo = PPP.randomHomog(lam)

resGP = newGP.rGP(pointpo.loc) 
Sigma = newGP.covMatrix(pointpo.loc)


locThin = np.empty((0,2))
valThin = np.empty((0,1))

    
locThin, valThin, Sigma = nthinSampler(25, newGP, locThin, valThin, pointpo.loc, resGP,Sigma)    
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
# Jitters one of the point process location and resamples the GP at said location
###

def locationMove(kappa, thisGP, locThin, valThin, locObs, valObs, Sigma):

    nlocThin = locThin.shape[0]
    
    moveInd = random.choice(np.array(range(0,nlocThin)))
    
    moveLoc = np.array([locThin[moveInd]])
    newLoc = np.array([[jitterBeta(moveLoc[:,0],kappa),jitterBeta(moveLoc[:,1],kappa)]])

    locTot = np.concatenate((locThin, locObs))
    valTot = np.concatenate((valThin, valObs))
    newVal, newSigma = thisGP.rCondGP1DSigma(newLoc, locTot, valTot, Sigma)
    
    acc_loc = kernelBeta(moveLoc, newLoc, kappa)/(1+np.exp(newVal))*(1+np.exp(valThin[moveInd]))/kernelBeta(newLoc, moveLoc, kappa) 
        
    U = random.uniform(size=1)
        
    if U < acc_loc:
        locThin = np.delete(locThin, moveInd, 0)
        valThin = np.delete(valThin, moveInd, 0)
        locThin = np.concatenate((newLoc, locThin))
        valThin = np.concatenate((newVal, valThin))
        Sigma = np.delete(np.delete(newSigma, moveInd+1, 0), moveInd+1, 1)

    return(locThin, valThin, Sigma)

### TESTER locationMove
kappa=10
newGP = GP(zeroMean,gaussianCov(1,1))

lam=4
pointpo = PPP.randomHomog(lam)

locThin = pointpo.loc
valThin = newGP.rGP(pointpo.loc) 

pointpo = PPP.randomHomog(lam)

locObs = pointpo.loc
valObs = newGP.rCondGP(locObs, locThin, valThin) 
Sigma = newGP.covMatrix(np.concatenate((locThin,locObs)))
print(locThin, valThin) 
    
locThinNew, valThinNew, Sigma = locationMove(kappa, newGP, locThin, valThin, locObs, valObs, Sigma)    
print(locThinNew, valThinNew) 
 


###
# Combines a mixture of birth-death-move type of samplers
###

def birthDeathMove(lam, kappa, thisGP, locThin, valThin, locObs, valObs, Sigma):
    
    nthin = locThin.shape[0]
    
    A = random.binomial(1,alpha(nthin),1)
    
    if A:
        locThin, valThin, Sigma = nthinSampler(lam, thisGP, locThin, valThin, locObs, valObs, Sigma)
    else:
        locThin, valThin, Sigma = locationMove(kappa, thisGP, locThin, valThin, locObs, valObs, Sigma)
    
    return(locThin, valThin, Sigma)

### TESTER birthDeathMove

kappa=10
newGP = GP(zeroMean,gaussianCov(1,0.1))

lam=4
pointpo = PPP.randomHomog(lam)

locThin = pointpo.loc
valThin = newGP.rGP(pointpo.loc) 

pointpo = PPP.randomHomog(lam)

locObs = pointpo.loc
valObs = newGP.rCondGP(locObs, locThin, valThin) 
Sigma = newGP.covMatrix(np.concatenate((locThin,locObs)))
print(locThin, valThin) 
    
locThinNew, valThinNew, Sigma = birthDeathMove(lam, kappa, newGP, locThin, valThin, locObs, valObs, Sigma)    
print(locThinNew, valThinNew)

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

###
# expit
###

def expit(x):
    return(np.exp(x)/(1+np.exp(x)))

### TESTER: expit
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr

expit(arr)


###
# potential energy in whitened space (AA^T=Sigma) Thinned events are first in Sigma
###

def U(whiteVal, A, nthin):
    
    return(np.sum(np.log(1+np.exp(-A[nthin:,:]@whiteVal))) + 
           np.sum(np.log(1+np.exp(A[:nthin,:]@whiteVal))) +
           1/2*np.transpose(whiteVal)@whiteVal)
    
### TESTER: U
newGP = GP(zeroMean,gaussianCov(1,1))

lam=10
pointpo = PPP.randomHomog(lam) 

locTot = pointpo.loc
Sigma = newGP.covMatrix(locTot)
valTot = newGP.rGP(pointpo.loc)

nthin = 3

A = np.linalg.cholesky(Sigma)
whiteVal = np.linalg.inv(A)@valTot

U(whiteVal,A,nthin)

###
# derivative of potential energy in whitened space (AA^T=Sigma) Thinned events are first in Sigma
###

def U_prime(whiteVal, A, nthin):

    return(-np.transpose(np.transpose(expit(-A[nthin:,:]@whiteVal))@A[nthin:,:]) +
           np.transpose(np.transpose(expit(A[:nthin,:]@whiteVal))@A[:nthin,:]) +
           whiteVal)

### TESTER: U_prime
newGP = GP(zeroMean,gaussianCov(1,1))

lam=10
pointpo = PPP.randomHomog(lam) 

locTot = pointpo.loc
Sigma = newGP.covMatrix(locTot)
valTot = newGP.rGP(pointpo.loc)

nthin = 3

A = np.linalg.cholesky(Sigma)
whiteVal = np.linalg.inv(A)@valTot

U_prime(whiteVal,A,nthin)

###
# sampling the GP values at the thinned and observed events
###

def functionSampler(delta,L,whiteVal,A,nthin):
    
    ntot = whiteVal.shape[0]
    
    v_init = random.normal(size=(ntot,1))
    
    v_prime = v_init - delta/2*U_prime(whiteVal, A, nthin)
    x_prime = whiteVal + delta*v_prime
    
    l=0
    while(l<L):
        v_prime = v_prime - delta*U_prime(x_prime,A,nthin)
        x_prime = x_prime + delta*v_prime
        
        l += 1
        
    v_prime = v_prime - delta/2*U_prime(x_prime,A,nthin)
    
    a_func = np.exp(-U(x_prime,A,nthin)+U(whiteVal,A,nthin)
                    - 1/2*np.transpose(v_prime)@v_prime
                    + 1/2*np.transpose(v_init)@v_init)
    
    Uf = random.uniform(size=1)
        
    if Uf < a_func:
        whiteVal = x_prime
    
    return(whiteVal)


### TESTER: functionSampler
newGP = GP(zeroMean,gaussianCov(1,1))

lam=10
pointpo = PPP.randomHomog(lam) 

locTot = pointpo.loc
Sigma = newGP.covMatrix(locTot)
valTot = newGP.rGP(pointpo.loc)

nthin = 3
delta = 0.1
L = 10

A = np.linalg.cholesky(Sigma)
whiteVal = np.linalg.inv(A)@valTot

valTot
whiteVal = functionSampler(delta,L,whiteVal,A,nthin)

A@whiteVal

### ###

newGP = GP(zeroMean,gaussianCov(1,1))

lam=10
pointpo = PPP.randomHomog(lam) 

locTot = np.array([[0.1,0.1],[0.2,0.2],[0.3,0.3],[0.8,0.8],[0.9,0.9]])
                    
Sigma = newGP.covMatrix(locTot)
valTot = newGP.rGP(locTot)

nthin = 3
delta = 0.1
L = 10

A = np.linalg.cholesky(Sigma)
whiteVal = np.linalg.inv(A)@valTot

valTot
whiteVal = functionSampler(delta,L,whiteVal,A,nthin)

A@whiteVal


###
# base intensity lambda sampler
###

def intensitySampler(mu,sigma2,ntot):
    alpha=mu**2/sigma2 + ntot
    beta=mu/sigma2 + 1
    lam = gamma.rvs(alpha, scale=1/beta)
    return(lam)


### TESTER: intensity sampler
mu = 100
sigma2 = 100

intensitySampler(mu,sigma2,500)

###
# full sampler of thinned locations, thinned values, observed values and intensity
###


def MCMCadams(size,lam_init,thisGP,thisPPP,nInsDelMov,kappa,delta,L,mu,sigma2):
    
    thinLoc = np.empty(shape=size,dtype=np.ndarray)
    thinVal = np.empty(shape=size,dtype=np.ndarray)
    obsVal = np.empty(shape=size,dtype=np.ndarray)
    lams = np.empty(shape=size)
    
    # initialization
    
    thinLoc[0] = PPP.randomHomog(lam=lam_init).loc
    nthin = thinLoc[0].shape[0]
    locTot = np.concatenate((thinLoc[0],thisPPP.loc))
    Sigma = thisGP.covMatrix(locTot)
    nloc = locTot.shape[0]
    
    totVal = rMultNorm(nloc,0,Sigma)
    
    thinVal[0] = totVal[0:nthin]
    obsVal[0] = totVal[nthin:nloc]
    lams[0] = lam_init
    
    i=1
    while i < size:
        locThin_prime, valThin_prime, Sigma = birthDeathMove(lams[i-1],kappa,thisGP,
                                                    thinLoc[i-1],thinVal[i-1],
                                                    thisPPP.loc,obsVal[i-1],Sigma)
        j=1
        while j < nInsDelMov:
            locThin_prime, valThin_prime, Sigma = birthDeathMove(lams[i-1],kappa,thisGP,
                                                    locThin_prime, valThin_prime,
                                                    thisPPP.loc,obsVal[i-1],Sigma)
            j+=1
        

        
        # locTot_prime = np.concatenate((locThin_prime,thisPPP.loc))
        valTot_prime = np.concatenate((valThin_prime,obsVal[i-1]))
        
        nthin = locThin_prime.shape[0]
        
        # Sigma = thisGP.covMatrix(locTot_prime)
        A = np.linalg.cholesky(Sigma)
        
        whiteVal_prime = np.linalg.inv(A)@valTot_prime
        
        whiteVal_prime = functionSampler(delta,L,whiteVal_prime,A,nthin)
        
        valTot_prime = A @ whiteVal_prime
        
        thinLoc[i] = locThin_prime
        thinVal[i] = valTot_prime[:nthin,:]
        obsVal[i] = valTot_prime[nthin:,:]
        
        ntot = valTot_prime.shape[0]
        
        lams[i] = intensitySampler(mu,sigma2,ntot)
        
        print(i)
        i+=1
    
    
    return(thinLoc,thinVal,obsVal,lams)

### TESTER: MCMCadams



def fct(x):
    return(np.exp((-x[:,0]**2-x[:,1]**2)/0.3))
# def fct(x):
#     return(np.exp(-(0.25-np.sqrt((x[:,0]-0.5)**2+(x[:,1]-0.5)**2))**2/0.003)*0.8+0.10)
# def fct(x):
#     return(np.exp(-np.minimum((x[:,0]-0.5)**2,(x[:,1]-0.5)**2)/0.007)*0.8+0.10)

lam_sim=1000

pointpo = PPP.randomNonHomog(lam_sim,fct)
pointpo.plot()


newGP = GP(zeroMean,gaussianCov(2,0.5))

niter=1000

import time

t0 = time.time()
thinLoc,thinVal,obsVal,lams = MCMCadams(niter,1000,newGP,pointpo,100,10,0.1,10,1000,1000)
t1 = time.time()

total1 = t1-t0

i=0
while(i < niter):


    

    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    
    plt.scatter(pointpo.loc[:,0],pointpo.loc[:,1])
    plt.scatter(thinLoc[i][:,0],thinLoc[i][:,1])
    

    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.show()
    
    
    fig.savefig("Scatter"+str(i)+".pdf", bbox_inches='tight')
    i+=1

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


res = 30
gridLoc = makeGrid([0,1], [0,1], res)



resGP = np.empty(shape=niter,dtype=np.ndarray)
i=0
t0 = time.time()
while(i < niter):
    resGP[i] = lams[i]*expit(newGP.rCondGP(gridLoc,np.concatenate((thinLoc[i],pointpo.loc)),
              np.concatenate((thinVal[i],obsVal[i]))))
    print(i)
    i+=1
t1 = time.time()

total2 = t1-t0

meanGP = np.mean(resGP)


#### to make plot ####

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
fig.savefig("meanInt.pdf", bbox_inches='tight')


#### plot of actual intensity ####

realInt = lam_sim*fct(gridLoc)

#### to make plot ####

imGP = np.transpose(realInt.reshape(res,res))

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
fig.savefig("trueInt.pdf", bbox_inches='tight')


#### first iteration ###



#### to make plot ####

imGP = np.transpose(resGP[0].reshape(res,res))

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
fig.savefig("initInt.pdf", bbox_inches='tight')


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.scatter(thinLoc[0][:,0],thinLoc[0][:,1])


plt.xlim(0,1)
plt.ylim(0,1)

plt.show()
fig.savefig("initScatter.pdf", bbox_inches='tight')

#### last iteration ###






#### to make plot ####

imGP = np.transpose(resGP[niter-1].reshape(res,res))

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
fig.savefig("finInt.pdf", bbox_inches='tight')


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.scatter(thinLoc[niter-1][:,0],thinLoc[niter-1][:,1])

plt.xlim(0,1)
plt.ylim(0,1)

plt.show()
fig.savefig("finScatter.pdf", bbox_inches='tight')



### every iteration ###

i=0
while(i < niter):


    
    imGP = np.transpose(resGP[i].reshape(res,res))
    
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
    fig.savefig("Int"+str(i)+".pdf", bbox_inches='tight')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    
    plt.scatter(pointpo.loc[:,0],pointpo.loc[:,1])
    plt.scatter(thinLoc[i][:,0],thinLoc[i][:,1])
    

    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.show()
    
    
    fig.savefig("Scatter"+str(i)+".pdf", bbox_inches='tight')
    i+=1

# ###
# # full sampler of thinned locations, thinned values, observed values and intensity
# ###


# def MCMCadams(size,lam_init,thisGP,thisPPP,nInsDel,kappa,delta,L,mu,sigma2):
    
#     thinLoc = np.empty(shape=size,dtype=np.ndarray)
#     thinVal = np.empty(shape=size,dtype=np.ndarray)
#     obsVal = np.empty(shape=size,dtype=np.ndarray)
#     lams = np.empty(shape=size)
    
#     # initialization
    
#     thinLoc[0] = PPP.randomHomog(lam=lam_init).loc
#     thinVal[0] = thisGP.rGP(thinLoc[0])
#     obsVal[0] = thisGP.rCondGP(thisPPP.loc,thinLoc[0],thinVal[0])
#     lams[0] = lam_init
    
#     i=1
#     while i < size:
#         locThin_prime, valThin_prime = nthinSampler(lams[i-1],thisGP,
#                                                     thinLoc[i-1],thinVal[i-1],
#                                                     thisPPP.loc,obsVal[i-1])
#         j=1
#         while j < nInsDel:
#             locThin_prime, valThin_prime = nthinSampler(lams[i-1],thisGP,
#                                                     locThin_prime, valThin_prime,
#                                                     thisPPP.loc,obsVal[i-1])
#             j+=1
        
#         # locThin_prime, valThin_prime = locationSampler(kappa,thisGP,
#         #                                                locThin_prime, valThin_prime,
#         #                                                thisPPP.loc,obsVal[i-1])
        
#         locTot_prime = np.concatenate((locThin_prime,thisPPP.loc))
#         valTot_prime = np.concatenate((valThin_prime,obsVal[i-1]))
        
#         nthin = locThin_prime.shape[0]
        
#         Sigma = thisGP.covMatrix(locTot_prime)
#         A = np.linalg.cholesky(Sigma)
        
#         whiteVal_prime = np.linalg.inv(A)@valTot_prime
        
#         whiteVal_prime = functionSampler(delta,L,whiteVal_prime,A,nthin)
        
#         valTot_prime = A @ whiteVal_prime
        
#         thinLoc[i] = locThin_prime
#         thinVal[i] = valTot_prime[:nthin,:]
#         obsVal[i] = valTot_prime[nthin:,:]
        
#         ntot = valTot_prime.shape[0]
        
#         lams[i] = intensitySampler(mu,sigma2,ntot)
        
#         i+=1
    
    
#     return(thinLoc,thinVal,obsVal,lams)

# ### TESTER: MCMCadams

# def fct(x):
#     return(np.exp((-x[:,0]**2-x[:,1]**2)/0.3))

# lam_sim=500

# pointpo = PPP.randomNonHomog(lam_sim,fct)
# pointpo.plot()


# newGP = GP(zeroMean,gaussianCov(1,1))

# niter=10

# import time

# t0 = time.time()
# thinLoc,thinVal,obsVal,lams = MCMCadams(niter,300,newGP,pointpo,10,10,0.1,10,300,10000)
# t1 = time.time()

# total1 = t1-t0


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


# t0 = time.time()
# resGP = expit(newGP.rCondGP(gridLoc,np.concatenate((thinLoc[9],pointpo.loc)),
#               np.concatenate((thinVal[9],obsVal[9]))))
# t1 = time.time()

# total2 = t1-t0



# #### to make plot ####

# imGP = np.transpose(resGP.reshape(res,res))

# x = np.linspace(0,1, res+1) 
# y = np.linspace(0,1, res+1) 
# X, Y = np.meshgrid(x,y) 

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_aspect('equal')

# plt.pcolormesh(X,Y,imGP)

# plt.xlim(0,1)
# plt.ylim(0,1)


# plt.show()


# i=0
# while(i < niter):
#     resGP = expit(newGP.rCondGP(gridLoc,np.concatenate((thinLoc[i],pointpo.loc)),
#               np.concatenate((thinVal[i],obsVal[i]))))

    
#     imGP = np.transpose(resGP.reshape(res,res))
    
#     x = np.linspace(0,1, res+1) 
#     y = np.linspace(0,1, res+1) 
#     X, Y = np.meshgrid(x,y) 
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_aspect('equal')
    
#     plt.pcolormesh(X,Y,imGP)
    
#     plt.xlim(0,1)
#     plt.ylim(0,1)
    
    
#     plt.show()
#     i+=1

# ### ###

# def fct(x):
#     return(np.exp(-(0.25-np.sqrt((x[:,0]-0.5)**2+(x[:,1]-0.5)**2))**2/0.003))

# lam_sim=500

# pointpo = PPP.randomNonHomog(lam_sim,fct)
# pointpo.plot()


# newGP = GP(zeroMean,gaussianCov(1,1))

# niter=10

# import time

# t0 = time.time()
# thinLoc,thinVal,obsVal,lams = MCMCadams(niter,300,newGP,pointpo,10,10,0.1,10,300,10000)
# t1 = time.time()

# total1 = t1-t0


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


# t0 = time.time()
# resGP = expit(newGP.rCondGP(gridLoc,np.concatenate((thinLoc[9],pointpo.loc)),
#               np.concatenate((thinVal[9],obsVal[9]))))
# t1 = time.time()

# total2 = t1-t0



# #### to make plot ####

# imGP = np.transpose(resGP.reshape(res,res))

# x = np.linspace(0,1, res+1) 
# y = np.linspace(0,1, res+1) 
# X, Y = np.meshgrid(x,y) 

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_aspect('equal')

# plt.pcolormesh(X,Y,imGP)

# plt.xlim(0,1)
# plt.ylim(0,1)


# plt.show()


# i=0
# while(i < niter):
#     resGP = expit(newGP.rCondGP(gridLoc,np.concatenate((thinLoc[i],pointpo.loc)),
#               np.concatenate((thinVal[i],obsVal[i]))))

    
#     imGP = np.transpose(resGP.reshape(res,res))
    
#     x = np.linspace(0,1, res+1) 
#     y = np.linspace(0,1, res+1) 
#     X, Y = np.meshgrid(x,y) 
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_aspect('equal')
    
#     plt.pcolormesh(X,Y,imGP)
    
#     plt.xlim(0,1)
#     plt.ylim(0,1)
    
    
#     plt.show()
#     i+=1


