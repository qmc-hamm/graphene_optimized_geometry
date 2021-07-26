# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:02:22 2021
theta1 = [21.78, 9.4, 7.3, 6.0, 5.1, 4.4, 3.89, 2.88, 2.0, 1.47, 1.16, 1.08, 1.05, 0.99, 0.5] 
@author: trakib2
"""

import numpy as np 
import h5py
import uuid

#print('input twist angle (Your possible options are-- \n 9.4, 7.3, 6.0, 4.4): \n')
#print a twist angle if you add your data 
def POSCAR_reader(filename):
    """
    input: POSCAR filename
    output: lattice vectors (3,3),
            atomic coordinates (N,3)
            number of atoms (int)
    """
    with open(filename,"r") as f:
        lines=f.readlines()

    latx_str = lines[2].split(' ')
    laty_str = lines[3].split(' ')
    latz_str = lines[4].split(' ')
    lat = np.zeros((3,3))
    
    for i in range (3):
        lat[0,i] = float(latx_str[i])
        lat[1,i] = float(laty_str[i])
        lat[2,i] = float(latz_str[i])
    
    atom_num = int(lines[6])
    coord = np.zeros((atom_num,3))
    
    for i in range (0, atom_num):
        coord_str = lines[i+8].split(' ')
        coord[i,0] = float(coord_str[0])
        coord[i,1] = float(coord_str[1])
        coord[i,2] = float(coord_str[2])
        
    return lat, coord, atom_num


if __name__=="__main__":
    with open('mn.txt', "r") as f:
        lines=f.readlines()
    t = np.zeros(len(lines))
    M = np.zeros(len(lines))
    N = np.zeros(len(lines))
    for i in range (len(lines)):
        tmn = lines[i].split(' ')
        t[i] = float(tmn[0])
        M[i] = int(tmn[1])
        N[i] = int(tmn[2])

    theta1 = [21.78, 9.4, 7.3, 6.0, 5.1, 4.4, 3.89, 2.88, 2.0, 1.47, 1.16, 1.08, 1.05, 0.99, 0.5] 
    for i in range(len(theta1)):
        theta = theta1[i]
        theta2 = str(theta1[i])
        st = theta2.split('.')
        thetaname = st[0]+'-'+st[1]
        method = 'Flat' #Input: method (String)
        ID = str(uuid.uuid4()) 
        date = 'May 15, 2021' #Input:date (String)
        
        filename = "raw/POSCAR_"+thetaname+"_hex.txt"
        lat, coord, atom_num = POSCAR_reader(filename)
        m = int(M[t==theta])
        n = int(N[t==theta])
        with h5py.File('best_estimate.hdf5', 'a') as hf:
            hf_ID = hf.create_group(ID)
            hf_ID.create_dataset('theta', data=theta)
            hf_ID.create_dataset('lattice_vector', data=lat)
            hf_ID.create_dataset('xyz', data=coord)
            hf_ID.create_dataset('m', data=m)
            hf_ID.create_dataset('n', data=n)
            hf_ID['method'] = method
            hf_ID['date'] = date
            #dt = h5py.special_dtype(vlen=str)
            #dset = hf_ID.create_dataset("method",(3,),dtype=dt)
            #dset.attrs["1"] = method
            #dset1 = hf_ID.create_dataset("date",(3,),dtype=dt)
            #dset1.attrs["2"] = date
            
            
            
        
            