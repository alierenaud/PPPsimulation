# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:37:17 2021

@author: alier
"""


import numpy as np


class dsymatrix:
    
    def __init__(self, size, arr, nObs):
        self.matrix = np.ndarray((size,size))
        self.ind = list(range(0,arr.shape[0]))
        self.indCem = list(range(arr.shape[0],size))
        self.matrix[np.ix_(self.ind,self.ind)] = arr
        self.inver = np.linalg.inv(arr)
        self.size = len(self.ind)
        self.nObs = nObs
        
    def sliceMatrix(self):
        return self.matrix[np.ix_(self.ind,self.ind)]
    
    # def size(self):
    #     return(len(self.ind))
    
    def delete(self, nb):
        
        nb += self.nObs
        
        ### updating inverse
        
        U = np.zeros(shape=(self.size,2))
        U[:,[1]] = self.matrix[np.ix_(self.ind,[self.ind[nb]])]
        U[nb,:] = [1,0]
        
        V = np.transpose(U)[::-1,:]
        
        SinvU = self.inver@U
        temp = self.inver + SinvU@np.linalg.inv(np.identity(2) - V@SinvU)@V@self.inver
        
        
        indmNb = [x for x in list(range(0,self.size)) if x !=nb]
        self.inver = temp[np.ix_(indmNb,indmNb)]
        
        self.indCem.insert(0,self.ind.pop(nb))
        self.size -= 1
        
    
    def concat(self,A,a):
        newInd = self.indCem.pop(0)
        AT = np.transpose(A)
        self.matrix[np.ix_([newInd],self.ind)] = AT
        self.matrix[np.ix_(self.ind,[newInd])] = A
        self.matrix[np.ix_([newInd],[newInd])] = a
        self.ind.append(newInd)
        self.size += 1
        
        ### updating inverse
        
        SA = self.inver@A
        s = 1/(a-AT@SA)
        SAT = np.transpose(SA)
        
        temp = np.ndarray((self.size,self.size))
        temp[np.ix_(list(range(0,self.size-1)),list(range(0,self.size-1)))] = self.inver + s*SA@SAT
        temp[np.ix_(list(range(0,self.size-1)),[-1])] = -s*SA
        temp[np.ix_([-1],list(range(0,self.size-1)))] = -s*SAT
        temp[-1,-1] = s
        
        self.inver = temp
    
    
    def change(self,A,i):
        
        i += self.nObs
        
        ### updating inverse
        
        U = np.zeros(shape=(self.size,2))
        U[:,[1]] = A - self.matrix[np.ix_(self.ind,[self.ind[i]])] 
        U[i,:] = [1,U[i,1]/2]
        
        V = np.transpose(U)[::-1,:]
        
        SinvU = self.inver@U
        self.inver = self.inver - SinvU@np.linalg.inv(np.identity(2) + V@SinvU)@V@self.inver
        
        self.matrix[np.ix_([self.ind[i]],self.ind)] = np.transpose(A)
        self.matrix[np.ix_(self.ind,[self.ind[i]])] = A
        
        
    
    def __repr__(self):
        return(self.sliceMatrix().__repr__())
    
    def __matmul__(self,other):
        return(self.sliceMatrix()@other)
   
n=50
eps=1/n
X = np.array([[0,eps/np.sqrt(1+eps**2),1/np.sqrt(1+eps**2)],[0,-eps/np.sqrt(1+eps**2),1/np.sqrt(1+eps**2)],[1/np.sqrt(2),0,-1/np.sqrt(2)]])
V = X@np.transpose(X)    
   

sigma = dsymatrix(100,V,0)


sigma@sigma.inver   

sigma.concat([[0],[0],[0]], 1) 
sigma
sigma@sigma.inver 

sigma.delete(0)
sigma
sigma@sigma.inver    


sigma.change([[0],[0],[1]], 2)
sigma
sigma@sigma.inver  

sigma = dsymatrix(100,np.array([[1,2,3],[4,5,6],[7,8,9]]),0)

sigma
sigma.ind
sigma.indCem

sigma.sliceMatrix()

sigma.delete(0)


sigma
sigma.ind
sigma.indCem

sigma.concat(np.array([[1],[0]]),1)

sigma

sigma.size

sigma.change([[1],[1],[1]], 2)

sigma

sigma@np.array([[1],[1],[1]])    
    



# class dmatrix:
    
#     def __init__(self, nrow, ncol, arr):
#         self.matrix = np.ndarray((nrow,ncol))
#         self.rowInd = list(range(0,arr.shape[0]))
#         self.colInd = list(range(0,arr.shape[1]))
#         self.rowIndCem = list(range(arr.shape[0],nrow))
#         self.colIndCem = list(range(arr.shape[1],ncol))
#         self.matrix[np.ix_(self.rowInd,self.colInd)] = arr
        
#     def sliceMatrix(self):
#         return self.matrix[np.ix_(self.rowInd,self.colInd)]
    
#     def nrow(self):
#         return(len(self.rowInd))
    
#     def ncol(self):
#         return(len(self.colInd))
    
#     def deleteRow(self, rowNb):
#         self.rowIndCem.insert(0,self.rowInd.pop(rowNb))
    
#     def deleteCol(self, colNb):
#         self.colIndCem.insert(0,self.colInd.pop(colNb))
        
#     def concatRow(self,C):
#         newRowInd = self.rowIndCem.pop(0)
#         self.rowInd.append(newRowInd)
#         self.matrix[np.ix_([newRowInd],self.colInd)] = C
    
#     def concatCol(self,B):
#         newColInd = self.colIndCem.pop(0)
#         self.colInd.append(newColInd)
#         self.matrix[np.ix_(self.rowInd,[newColInd])] = B
        
    
#     def __repr__(self):
#         return(self.sliceMatrix().__repr__())
    
#     def __matmul__(self,other):
#         return(self.sliceMatrix()@other)
    
    
# sigma = dmatrix(100,100, np.array([[1,2,3],[4,5,6],[7,8,9]]))


# sigma
# sigma.rowInd
# sigma.colInd
# sigma.rowIndCem
# sigma.colIndCem

# sigma.sliceMatrix()

# sigma.deleteRow(0)
# sigma.deleteCol(0)


# sigma
# sigma.rowInd
# sigma.colInd
# sigma.rowIndCem
# sigma.colIndCem

# sigma.concatRow(np.array([[1,0]]))
# sigma.concatCol(np.array([[1],[0],[8]]))

# sigma

# sigma.nrow()
# sigma.ncol()

# sigma@np.array([[1],[1],[1]])



# #### race
# import time


# N = 10000

# from numpy import random

# initSize = 1000
# random.seed(0)

# arr = random.normal(size=(initSize, initSize))


# dynArr = dmatrix(5000,5000, arr)


# i=0
# t0 = time.time()
# while i<N:
    
#     ind = random.randint(dynArr.nrow())
#     dynArr.deleteRow(ind)
    
#     ind = random.randint(dynArr.ncol())
#     dynArr.deleteCol(ind)
    
#     newRow = random.normal(size=(1,dynArr.ncol()))
#     dynArr.concatRow(newRow)
    
#     newCol = random.normal(size=(dynArr.nrow(),1))
#     dynArr.concatCol(newCol)
    
#     i+=1
    
# t1 = time.time()

# total1 = t1-t0

# random.seed(0)

# arr = random.normal(size=(initSize, initSize))

# i=0
# t0 = time.time()
# while i<N:
    
#     ind = random.randint(arr.shape[0])
#     arr = np.delete(arr, ind, 0)
    
#     ind = random.randint(arr.shape[1])
#     arr = np.delete(arr, ind, 1)
    
#     newRow = random.normal(size=(1,arr.shape[1]))
#     arr = np.concatenate((arr,newRow),0)
    
#     newCol = random.normal(size=(arr.shape[0],1))
#     arr = np.concatenate((arr,newCol),1)
    
#     i+=1

# t1 = time.time()

# total2= t1-t0

# print(total1)
# print(total2)


### data structure for thinned locations


class bdmatrix:
    
    
    def __init__(self, nrow, arr, nObs, size):
        ncol = arr.shape[1]
        self.length = arr.shape[0]
        self.nObs = nObs
        self.nThin = self.length - nObs
        self.matrix = np.ndarray((nrow,ncol))
        self.indTable = np.empty(shape=size,dtype=list)
        self.indTable[0] = list(range(0,self.length))
        self.matrix[np.ix_(list(range(0,self.length)),list(range(0,ncol)))] = arr
        self.sampNb=0
      

        
        
    def nextSamp(self):
        self.indTable[self.sampNb+1] = self.indTable[self.sampNb].copy()
        self.sampNb +=1
        
    def newVals(self,A):
        
        newLength = self.length + self.nObs + self.nThin
        self.indTable[self.sampNb] = list(range(self.length, newLength))
        self.matrix[self.indTable[self.sampNb]] = A
        
        
        
        self.length=newLength
        
    def birth(self,loc):
        self.indTable[self.sampNb].append(self.length)
        self.matrix[self.length]=loc
        self.length +=1
        self.nThin += 1
        
    def death(self,i):
        self.indTable[self.sampNb].pop(i+self.nObs)
        self.nThin -= 1
        
    def move(self,i,loc):
        self.indTable[self.sampNb][i+self.nObs] = self.length
        self.matrix[self.length]=loc
        self.length +=1
        
    def getThinLoc(self,ind):
        return self.matrix[self.indTable[self.sampNb][self.nObs+ind],:]
        
    def obsLoc(self):
        return self.matrix[self.indTable[self.sampNb][:self.nObs],:]
    
    def thinLoc(self):
        return self.matrix[self.indTable[self.sampNb][self.nObs:],:]
    
    def totLoc(self):
        return self.matrix[self.indTable[self.sampNb],:]
        
        
        
    # def __getitem__(self, items):
    #     if isinstance(items, int):
    #         return([self.matrix[i,:] for i in [self.indTable[items]]])
    #     else:
    #         return([self.matrix[i,:] for i in self.indTable[items]])
            
        
        

### init

arr= np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])        
        
newBDM = bdmatrix(10,arr,2,5)        
        
newBDM.indTable        
newBDM.obsLoc() 
newBDM.thinLoc()
newBDM.nThin
newBDM.totLoc()       
        


### nextSamp

newBDM.nextSamp()
newBDM.indTable        
newBDM.obsLoc() 
newBDM.thinLoc()
newBDM.nThin
newBDM.totLoc()   

## birth

newBDM.birth([7,8])

## death

newBDM.death(1)

newBDM.indTable        
newBDM.obsLoc() 
newBDM.thinLoc()
newBDM.nThin
newBDM.totLoc()   

newBDM.nextSamp()
newBDM.death(2)
newBDM.birth([9,10])

## move

newBDM.move(0, [11,12])


newBDM.indTable        
newBDM.obsLoc() 
newBDM.thinLoc()
newBDM.nThin
newBDM.getThinLoc(0)
newBDM.getThinLoc(1)
newBDM.getThinLoc(2)
newBDM.totLoc()   


newBDM.death(0)
newBDM.death(0)
newBDM.thinLoc()
newBDM.nThin


newBDM.matrix




# ### getitem      
        
# newBDM[0]        
# newBDM[[0,1]]        
# newBDM[[0,1,2]]        
        
# newBDM.indTable        
# newBDM.matrix         
        
        

### Test as container for values


arr= np.array([[1],[3],[5],[7],[9]])   

newBDM = bdmatrix(50,arr,2,5)    



newBDM.death(0)
newBDM.death(1)

newBDM.thinLoc()

newBDM.move(0, 40)
newBDM.thinLoc()


newBDM.birth(56)
newBDM.nThin
newBDM.birth(44)
newBDM.thinLoc()

newBDM.nextSamp()
newBDM.indTable
newBDM.totLoc()


newBDM.newVals([[10],[20],[1],[2],[3]])
newBDM.thinLoc()
newBDM.indTable
newBDM.matrix



newBDM.death(2)
newBDM.move(1, 7)
newBDM.nThin
newBDM.thinLoc()
newBDM.obsLoc()
newBDM.totLoc()


newBDM.nextSamp()
newBDM.indTable



