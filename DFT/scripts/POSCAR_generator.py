#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 13:56:02 2021

@author: tawfiqurrakib
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:02:22 2021

@author: trakib2
"""

import numpy as np 

def POSCAR_writer(filename, atom_num, a1, a2, b2, c3, xyz):
    '''
    Parameters
    ----------
    input: 
    filename: POSCAR filename
    atom_num : Number of atoms
        
    a1 : lattice constant, lat[0,0]
    a2 : lattice constant, lat[1,0]
    b2 : lattice constant, lat[1,1]
    c3 : lattice constant, lat[2,2]
    xyz : Coordinates

    Returns
    -------
    file

    '''
    f = open(filename, "w")
    f.write('written by TR \n')
    f.write('%.9f \n'%(1.0))
    f.write('%.9f %.9f %.9f \n'%(a1, 0.0, 0.0))
    f.write('%.9f %.9f %.9f \n'%(a2,b2, 0.0))
    f.write('%.9f %.9f %.9f \n'%(0.0, 0.0, c3))
    f.write('Type1 \n')
    f.write('%d\n'%(atom_num))
    f.write('Cartesian \n')

    for i in range (atom_num):
        f.write('%1.8f %1.8f %1.8f \n'%(xyz[i,0],xyz[i,1],xyz[i,2]))

    
    f.close()

def lammps_data_reader(filename):
    '''
    

    Parameters
    ----------
    input: 
    filename: lammps filename

    Returns
    -------
    atom_num : number of atoms
    lenx : length along x-direction
    leny : length along y-direction
    lenz : length along z-direction
    xyz : coordinates

    '''
    f=open(filename, "r")
    lines=f.readlines()
    
    atom_num = int(lines[2].split('\t')[0])
    lenx = float(lines[4].split('\t')[1])
    leny = float(lines[5].split('\t')[1])
    lenz = float(lines[6].split('\t')[1])
    
    xyz = np.zeros((atom_num,3))
    for i in range (atom_num):
        xyz[i,0] = float(lines[i+16].split(' ')[4])
        xyz[i,1] = float(lines[i+16].split(' ')[5])
        xyz[i,2] = float(lines[i+16].split(' ')[6])
    f.close()
    return atom_num, lenx, leny, lenz, xyz


if __name__=="__main__":
    filename = "structure_lammps.txt";
    atom_num, lenx, leny, lenz, xyz = lammps_data_reader(filename)
    filename1 = "POSCAR";
    a1 = lenx
    a2 = 0.0
    b2 = leny
    c3 = lenz
    POSCAR_writer(filename1, atom_num, a1, a2, b2, c3, xyz)





