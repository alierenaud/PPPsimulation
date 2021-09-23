# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 21:09:36 2021

@author: alier
"""

from rppp import PPP
import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from rGP import GP
from rGP import zeroMean
from rGP import expCov

from rppp import mtPPP
from scipy.stats import matrix_normal
from gfunc_est import gfuncest


### circle ############################

lam=500


pointpo = PPP.randomHomog(lam)

def fct(x):
    return(np.exp(-(0.25-np.sqrt((x[:,0]-0.5)**2+(x[:,1]-0.5)**2))**2/0.005))

index = np.greater(fct(pointpo.loc),random.uniform(size=pointpo.loc.shape[0]))
nhPPP = pointpo.loc[index]



fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

# plt.plot(pointpo.loc[:,0],pointpo.loc[:,1], 'o')
plt.plot(nhPPP[:,0],nhPPP[:,1], 'o')
plt.xlim(0,1)
plt.ylim(0,1)

plt.show()
fig.savefig("circle.pdf", bbox_inches='tight')  

### corner ############################


lam=500


pointpo = PPP.randomHomog(lam)

def fct(x):
    return(np.exp(-((x[:,0])+(x[:,1]))**2/0.3))

index = np.greater(fct(pointpo.loc),random.uniform(size=pointpo.loc.shape[0]))
nhPPP = pointpo.loc[index]



fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

# plt.plot(pointpo.loc[:,0],pointpo.loc[:,1], 'o')
plt.plot(nhPPP[:,0],nhPPP[:,1], 'o')
plt.xlim(0,1)
plt.ylim(0,1)

plt.show()
fig.savefig("corner.pdf", bbox_inches='tight')  


### SGCD #########




lam=500
tau=1/4
rho=5

def expit(x):
    return(np.exp(x)/(1+np.exp(x)))




locs = PPP.randomHomog(lam).loc

newGP = GP(zeroMean,expCov(tau,rho))
GPvals = newGP.rGP(locs)

locProb  = expit(GPvals)
index = np.array(np.greater(locProb,random.uniform(size=locProb.shape)))
locObs = locs[np.squeeze(index)]


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

# plt.plot(pointpo.loc[:,0],pointpo.loc[:,1], 'o')
plt.plot(locObs[:,0],locObs[:,1], 'o')
plt.xlim(0,1)
plt.ylim(0,1)

plt.show()
fig.savefig("1DSGCD.pdf", bbox_inches='tight')  


### Homog PPP with int ###########


lam=500


pointpo = PPP.randomHomog(lam)

def cst(x):
    return(np.array([1/2 for i in x]))

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


res =50
gridLoc = makeGrid([0,1], [0,1], res)
gridInt = cst(gridLoc)*lam

index = np.greater(cst(pointpo.loc),random.uniform(size=pointpo.loc.shape[0]))
nhPPP = pointpo.loc[index]


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

imGP = np.transpose(gridInt.reshape(res,res))

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 
    

ax.set_aspect('equal')
    
ff = ax.pcolormesh(X,Y,imGP, cmap='gray', vmin=0, vmax=500)

fig.colorbar(ff)   

plt.scatter(nhPPP[:,0],nhPPP[:,1], c="tab:orange")

plt.show()
fig.savefig("HPPP.pdf", bbox_inches='tight') 

### Non-Homog PPP with int ###########

lam=500


pointpo = PPP.randomHomog(lam)

def fct(x):
    return(np.exp(-np.minimum((x[:,0]-0.5)**2,(x[:,1]-0.5)**2)/0.01))


gridInt = fct(gridLoc)*lam

index = np.greater(fct(pointpo.loc),random.uniform(size=pointpo.loc.shape[0]))
nhPPP = pointpo.loc[index]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

imGP = np.transpose(gridInt.reshape(res,res))

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 
    

ax.set_aspect('equal')
    
ff = ax.pcolormesh(X,Y,imGP, cmap='gray', vmin=0, vmax=500)

fig.colorbar(ff)   

plt.scatter(nhPPP[:,0],nhPPP[:,1], c="tab:orange")

plt.show()
fig.savefig("NHPPP.pdf", bbox_inches='tight') 



### Homog PP before thin ###########

lam=500


pointpo = PPP.randomHomog(lam)

def fct(x):
    return(np.exp(-np.minimum((x[:,0]-0.5)**2,(x[:,1]-0.5)**2)/0.01))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')


plt.scatter(pointpo.loc[:,0],pointpo.loc[:,1])


plt.xlim(0,1)
plt.ylim(0,1)


plt.show()
fig.savefig("HomogBefThin.pdf", bbox_inches='tight') 


### Non Homog after thin ###########


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

index = np.greater(fct(pointpo.loc),random.uniform(size=pointpo.loc.shape[0]))
nhPPP = pointpo.loc[index]
thinlocs = pointpo.loc[~index]


plt.scatter(thinlocs[:,0],thinlocs[:,1], c=(0.75, 0.75, 0.75))
plt.scatter(nhPPP[:,0],nhPPP[:,1])

plt.xlim(0,1)
plt.ylim(0,1)

plt.show()
fig.savefig("NonHomogAfThin.pdf", bbox_inches='tight') 


### SGCD with intensity #####



lam=500
tau=1/4
rho=5

def expit(x):
    return(np.exp(x)/(1+np.exp(x)))

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


locs = PPP.randomHomog(lam).loc

newGP = GP(zeroMean,expCov(tau,rho))
GPvals = newGP.rGP(np.concatenate((locs,gridLoc)))


gridInt = lam*expit(GPvals[locs.shape[0]:,:])


locProb  = expit(GPvals[:locs.shape[0],:])
index = np.array(np.greater(locProb,random.uniform(size=locProb.shape)))
locObs = locs[np.squeeze(index)]
# locThin = locs[np.logical_not(np.squeeze(index))]




# ### cox process


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_aspect('equal')

# plt.xlim(0,1)
# plt.ylim(0,1)

# plt.scatter(locObs[:,0],locObs[:,1])

# plt.show()

### int + obs + thin

imGP = np.transpose(gridInt.reshape(res,res))

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.pcolormesh(X,Y,imGP, cmap="gray")

plt.xlim(0,1)
plt.ylim(0,1)
plt.colorbar()

plt.scatter(locObs[:,0],locObs[:,1], color= "tab:orange")
# plt.scatter(locThin[:,0],locThin[:,1], color= "white", s=1)


plt.show()
fig.savefig("SGCPwInt.pdf", bbox_inches='tight') 


#### mtSGCD Examples


## Generate a mtype SGCD

lam=500
rho=5

locs = PPP.randomHomog(lam).loc


newGP = GP(zeroMean,expCov(1,rho))

U = newGP.covMatrix(locs)

var=4

# V = np.array([[1]])*var
V = np.array([[1,-0.9],[-0.9,1]])*var
# V = np.array([[1,0.9],[0.9,1]])*var
# V = np.array([[1,0.5],[0.5,1]])*var
# V = np.array([[1,-4.5],[-4.5,25]])*var

# mu=-2
mu=np.array([-2,-3])

X = matrix_normal.rvs(rowcov=U, colcov=V) + mu

def multExpit(x):
    N = np.sum(np.exp(x))
    probs = np.array([np.exp(i)/(1+N) for i in x])
    return(np.append(probs,1-np.sum(probs)))
        
        
probs = np.array([multExpit(x) for x in X])

nch = probs.shape[1]

colours = np.array([np.random.choice(nch,p=p) for p in probs])

locs1 = locs[colours == 0]
locs2 = locs[colours == 1]





### make an mtPP format object


pp1 = PPP(locs1)
pp2 = PPP(locs2)

# pps = np.array([pp1])
pps = np.array([pp1,pp2])

mtppR = mtPPP(pps)

mtppR.plot()


#### est g function

N = 10000




rho = 5


steps = np.linspace(0.0, 0.8, num=50)
gsR = gfuncest(N,V,mu,rho,steps)

plt.plot(steps,gsR[:,0])
# plt.show()
plt.plot(steps,gsR[:,2])
# plt.show()
plt.plot(steps,gsR[:,1],  c="grey")
plt.show()


### Attractive example


## Generate a mtype SGCD

lam=500
rho=5

locs = PPP.randomHomog(lam).loc


newGP = GP(zeroMean,expCov(1,rho))

U = newGP.covMatrix(locs)

var=4

# V = np.array([[1]])*var
# V = np.array([[1,-0.9],[-0.9,1]])*var
V = np.array([[1,0.9],[0.9,1]])*var
# V = np.array([[1,0.5],[0.5,1]])*var
# V = np.array([[1,-4.5],[-4.5,25]])*var

# mu=-2
mu=np.array([-2,-3])

X = matrix_normal.rvs(rowcov=U, colcov=V) + mu

def multExpit(x):
    N = np.sum(np.exp(x))
    probs = np.array([np.exp(i)/(1+N) for i in x])
    return(np.append(probs,1-np.sum(probs)))
        
        
probs = np.array([multExpit(x) for x in X])

nch = probs.shape[1]

colours = np.array([np.random.choice(nch,p=p) for p in probs])

locs1 = locs[colours == 0]
locs2 = locs[colours == 1]





### make an mtPP format object


pp1 = PPP(locs1)
pp2 = PPP(locs2)

# pps = np.array([pp1])
pps = np.array([pp1,pp2])

mtppA = mtPPP(pps)

mtppA.plot()


#### est g function

N = 10000




rho = 5


steps = np.linspace(0.0, 0.8, num=50)
gsA = gfuncest(N,V,mu,rho,steps)

plt.plot(steps,gsA[:,0])
# plt.show()
plt.plot(steps,gsA[:,2])
# plt.show()
plt.plot(steps,gsA[:,1],  c="grey")
plt.show()


### Neutral example


## Generate a mtype SGCD

lam=500
rho=5

locs = PPP.randomHomog(lam).loc


newGP = GP(zeroMean,expCov(1,rho))

U = newGP.covMatrix(locs)

var=4

# V = np.array([[1]])*var
# V = np.array([[1,-0.9],[-0.9,1]])*var
# V = np.array([[1,0.9],[0.9,1]])*var
V = np.array([[1,0.5],[0.5,1]])*var
# V = np.array([[1,-4.5],[-4.5,25]])*var

# mu=-2
mu=np.array([-2,-3])

X = matrix_normal.rvs(rowcov=U, colcov=V) + mu

def multExpit(x):
    N = np.sum(np.exp(x))
    probs = np.array([np.exp(i)/(1+N) for i in x])
    return(np.append(probs,1-np.sum(probs)))
        
        
probs = np.array([multExpit(x) for x in X])

nch = probs.shape[1]

colours = np.array([np.random.choice(nch,p=p) for p in probs])

locs1 = locs[colours == 0]
locs2 = locs[colours == 1]





### make an mtPP format object


pp1 = PPP(locs1)
pp2 = PPP(locs2)

# pps = np.array([pp1])
pps = np.array([pp1,pp2])

mtppN = mtPPP(pps)

mtppN.plot()


#### est g function

N = 10000




rho = 5


steps = np.linspace(0.0, 0.8, num=50)
gsN = gfuncest(N,V,mu,rho,steps)

plt.plot(steps,gsN[:,0])
# plt.show()
plt.plot(steps,gsN[:,2])
# plt.show()
plt.plot(steps,gsN[:,1],  c="grey")
plt.show()


### 6x6 plot



fig, axs = plt.subplots(2, 3, figsize=(10,7))

fig.suptitle("Pair Correlation Functions")


### Repulsive

axs[0,0].title.set_text("Repulsive")

axs[0,0].plot(steps,gsR[:,0])
# plt.show()
axs[0,0].plot(steps,gsR[:,2])
# plt.show()
axs[0,0].plot(steps,gsR[:,1],  c="grey")

axs[0,0].set_xlabel("Distance")


axs[1,0].set_aspect('equal')


axs[1,0].set_xlim(0,1)
axs[1,0].set_ylim(0,1)

axs[1,0].scatter(mtppR.pps[0].loc[:,0],mtppR.pps[0].loc[:,1])
axs[1,0].scatter(mtppR.pps[1].loc[:,0],mtppR.pps[1].loc[:,1])



### Attractive

axs[0,1].title.set_text("Attractive")

axs[0,1].plot(steps,gsA[:,0])
# plt.show()
axs[0,1].plot(steps,gsA[:,2])
# plt.show()
axs[0,1].plot(steps,gsA[:,1],  c="grey")

axs[0,1].set_xlabel("Distance")


axs[1,1].set_aspect('equal')


axs[1,1].set_xlim(0,1)
axs[1,1].set_ylim(0,1)

axs[1,1].scatter(mtppA.pps[0].loc[:,0],mtppA.pps[0].loc[:,1])
axs[1,1].scatter(mtppA.pps[1].loc[:,0],mtppA.pps[1].loc[:,1])


### Neutral

axs[0,2].title.set_text("Neutral")

axs[0,2].plot(steps,gsN[:,0])
# plt.show()
axs[0,2].plot(steps,gsN[:,2])
# plt.show()
axs[0,2].plot(steps,gsN[:,1],  c="grey")

axs[0,2].set_xlabel("Distance")


axs[1,2].set_aspect('equal')


axs[1,2].set_xlim(0,1)
axs[1,2].set_ylim(0,1)

axs[1,2].scatter(mtppN.pps[0].loc[:,0],mtppN.pps[0].loc[:,1])
axs[1,2].scatter(mtppN.pps[1].loc[:,0],mtppN.pps[1].loc[:,1])




plt.show()
fig.savefig("SGCPexamples.pdf", bbox_inches='tight') 



