# -*- coding: utf-8 -*-
"""
Created on Sat May 15 18:27:00 2021

@author: trakib2
"""

import numpy as np 
import h5py

def find_angle(f, angle, tol):
    identity = []
    for k in f.keys():
        if np.isclose(f[f"{k}/theta"],angle, atol = tol):
            identity = np.append(identity, k)
    
    return identity      

tol = 0.05
angle = 0.98
f = h5py.File('best_estimate.hdf5', 'r')

       
ID = find_angle(f, angle, tol)
print(ID)
