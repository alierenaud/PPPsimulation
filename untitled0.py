# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 12:21:16 2021

@author: alier
"""

import numpy as np

arr = np.array([[1,2,3],[4,5,6],[7,8,9]])


arr2 = arr[:2]

arr2[0] = [7,8,9]

arr



arr = np.array([[1,2,3],[4,5,6],[7,8,9]])


arr2 = arr[1:3,1:3]

arr2[0,:] = [7,8]

arr



arr = np.array([[1,2,3],[4,5,6],[7,8,9]])


arr2 = arr[np.ix_([0,2],[0,2])]

arr2[0,:] = [7,8]

arr


arr = np.array([[1,2,3],[4,5,6],[7,8,9]])

arr2 = arr[:2,2]

arr2[0]=4

arr

arr[:2,:2] = arr[1:,1:]

arr2


from numpy import random
n=10

arr= random.normal(size=(n,n))

arr[:(n-1),:(n-1)] = arr[1:,1:]


np.savetxt("randNorm.csv",arr,delimiter=",")

arr2 = np.loadtxt("randNorm.csv",delimiter=",")
type(arr2)
