# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:37:17 2021

@author: alier
"""

# this tha new branch peeps

import numpy as np


class dsymatrix:
    
    def __init__(self, size, arr):
        self.matrix = np.ndarray((size,size))
        self.ind = list(range(0,arr.shape[0]))
        self.indCem = list(range(arr.shape[0],size))
        self.matrix[np.ix_(self.ind,self.ind)] = arr
        
    def sliceMatrix(self):
        return self.matrix[np.ix_(self.ind,self.ind)]
    
    def size(self):
        return(len(self.ind))
    
    def delete(self, nb):
        self.indCem.insert(0,self.ind.pop(nb))
    
    def concat(self,A):
        newInd = self.indCem.pop(0)
        self.ind.append(newInd)
        self.matrix[np.ix_([newInd],self.ind)] = A
        self.matrix[np.ix_(self.ind,[newInd])] = np.transpose(A)       
    
    def __repr__(self):
        return(self.sliceMatrix().__repr__())
    
    def __matmul__(self,other):
        return(self.sliceMatrix()@other)
    
sigma = dsymatrix(100,np.array([[1,2,3],[4,5,6],[7,8,9]]))


sigma
sigma.ind
sigma.indCem

sigma.sliceMatrix()

sigma.delete(0)


sigma
sigma.ind
sigma.indCem

sigma.concat(np.array([[1,0,1]]))

sigma

sigma.size()

sigma@np.array([[1],[1],[1]])    
    



class dmatrix:
    
    def __init__(self, nrow, ncol, arr):
        self.matrix = np.ndarray((nrow,ncol))
        self.rowInd = list(range(0,arr.shape[0]))
        self.colInd = list(range(0,arr.shape[1]))
        self.rowIndCem = list(range(arr.shape[0],nrow))
        self.colIndCem = list(range(arr.shape[1],ncol))
        self.matrix[np.ix_(self.rowInd,self.colInd)] = arr
        
    def sliceMatrix(self):
        return self.matrix[np.ix_(self.rowInd,self.colInd)]
    
    def nrow(self):
        return(len(self.rowInd))
    
    def ncol(self):
        return(len(self.colInd))
    
    def deleteRow(self, rowNb):
        self.rowIndCem.insert(0,self.rowInd.pop(rowNb))
    
    def deleteCol(self, colNb):
        self.colIndCem.insert(0,self.colInd.pop(colNb))
        
    def concatRow(self,C):
        newRowInd = self.rowIndCem.pop(0)
        self.rowInd.append(newRowInd)
        self.matrix[np.ix_([newRowInd],self.colInd)] = C
    
    def concatCol(self,B):
        newColInd = self.colIndCem.pop(0)
        self.colInd.append(newColInd)
        self.matrix[np.ix_(self.rowInd,[newColInd])] = B
        
    
    def __repr__(self):
        return(self.sliceMatrix().__repr__())
    
    def __matmul__(self,other):
        return(self.sliceMatrix()@other)
    
    
sigma = dmatrix(100,100, np.array([[1,2,3],[4,5,6],[7,8,9]]))


sigma
sigma.rowInd
sigma.colInd
sigma.rowIndCem
sigma.colIndCem

sigma.sliceMatrix()

sigma.deleteRow(0)
sigma.deleteCol(0)


sigma
sigma.rowInd
sigma.colInd
sigma.rowIndCem
sigma.colIndCem

sigma.concatRow(np.array([[1,0]]))
sigma.concatCol(np.array([[1],[0],[8]]))

sigma

sigma.nrow()
sigma.ncol()

sigma@np.array([[1],[1],[1]])



#### race
import time


N = 10000

from numpy import random

initSize = 1000
random.seed(0)

arr = random.normal(size=(initSize, initSize))


dynArr = dmatrix(5000,5000, arr)


i=0
t0 = time.time()
while i<N:
    
    ind = random.randint(dynArr.nrow())
    dynArr.deleteRow(ind)
    
    ind = random.randint(dynArr.ncol())
    dynArr.deleteCol(ind)
    
    newRow = random.normal(size=(1,dynArr.ncol()))
    dynArr.concatRow(newRow)
    
    newCol = random.normal(size=(dynArr.nrow(),1))
    dynArr.concatCol(newCol)
    
    i+=1
    
t1 = time.time()

total1 = t1-t0

random.seed(0)

arr = random.normal(size=(initSize, initSize))

i=0
t0 = time.time()
while i<N:
    
    ind = random.randint(arr.shape[0])
    arr = np.delete(arr, ind, 0)
    
    ind = random.randint(arr.shape[1])
    arr = np.delete(arr, ind, 1)
    
    newRow = random.normal(size=(1,arr.shape[1]))
    arr = np.concatenate((arr,newRow),0)
    
    newCol = random.normal(size=(arr.shape[0],1))
    arr = np.concatenate((arr,newCol),1)
    
    i+=1

t1 = time.time()

total2= t1-t0

print(total1)
print(total2)










