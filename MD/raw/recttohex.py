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
    with open(filename,"r") as f:
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

    return atom_num, lenx, leny, lenz, xyz

def periodic_extension(atom_num, lenx, leny, lenz, px, py, xyz):
    '''
    

    Parameters
    ----------
    atom_num : number of atoms
    lenx : length along x-direction
    leny : length along y-direction
    lenz : length along z-direction
    xyz : coordinates
    px : Repitition along x-direction
    py : Repitition along y-direction


    Returns
    -------
    periodic_atom_num : Number of atoms in extended cell
    periodic_lenx : length along x-direction of new cell
    periodic_leny : length along y-direction of new cell
    periodic_lenz : length along z-direction of new cell
    periodic_xyz : coordinates of new cell

    '''
    periodic_lenx = lenx*px
    periodic_leny = leny*py
    periodic_lenz = lenz
    periodic_xyz = np.zeros((atom_num*px*py,3))
    periodic_xyz[0:atom_num,0] = xyz[0:atom_num,0] 
    periodic_xyz[0:atom_num,1] = xyz[0:atom_num,1] 
    periodic_xyz[0:atom_num,2] = xyz[0:atom_num,2] 
    if px>1:
        for i in range (1,px):
            periodic_xyz[i*atom_num:(i+1)*atom_num,0] = periodic_xyz[(i-1)*atom_num:i*atom_num,0] + lenx
            periodic_xyz[i*atom_num:(i+1)*atom_num,1] = periodic_xyz[(i-1)*atom_num:i*atom_num,1] 
            periodic_xyz[i*atom_num:(i+1)*atom_num,2] = periodic_xyz[(i-1)*atom_num:i*atom_num,2] 
            
    if py>1:
        for i in range (1,py):
            periodic_xyz[i*atom_num*px:(i+1)*atom_num*px,0] = periodic_xyz[(i-1)*atom_num*px:i*atom_num*px,0] 
            periodic_xyz[i*atom_num*px:(i+1)*atom_num*px,1] = periodic_xyz[(i-1)*atom_num*px:i*atom_num*px,1] + leny
            periodic_xyz[i*atom_num*px:(i+1)*atom_num*px,2] = periodic_xyz[(i-1)*atom_num*px:i*atom_num*px,2]
    periodic_atom_num = atom_num*px*py
    return periodic_atom_num, periodic_lenx, periodic_leny, periodic_lenz, periodic_xyz

def recttohex_cutter(atom_num, periodic_atom_num, periodic_lenx, periodic_leny, periodic_lenz, periodic_xyz, tol):
    '''
    cuts the rectangular cell to hexagonal cell
    
    Parameters
    ----------
    
    
    periodic_atom_num : Number of atoms in extended cell
    periodic_lenx : length along x-direction of new cell
    periodic_leny : length along y-direction of new cell
    periodic_lenz : length along z-direction of new cell
    periodic_xyz : coordinates of new cell
    tol :Tolerance
        DESCRIPTION.

    Returns
    -------
    hex_atom_num : Number of atoms in hexagonal cell
    a1 : lattice constant, lat[0,0]
    a2 : lattice constant, lat[1,0]
    b2 : lattice constant, lat[1,1]
    c3 : lattice constant, lat[2,2]
    hex_xyz : coordinates of hexagonal cell

    '''
    angle = 60*np.pi/180
    a1 = periodic_lenx/2
    a2 = a1/2
    b2 = (periodic_lenx*np.cos(angle/2))/2
    c3 = periodic_lenz
    hex_xyz = np.zeros((int(atom_num/2),3))
    hex_atom_num = int(atom_num/2)
    p = 0
    for i in range (periodic_atom_num):
        if (periodic_xyz[i,1]<periodic_leny/2-tol):
            if (periodic_xyz[i,0]>periodic_xyz[i,1]/np.tan(angle)-tol) and \
                (periodic_xyz[i,0]<periodic_lenx/2 + periodic_xyz[i,1]/np.tan(angle)-tol):
                    hex_xyz[p,0] = periodic_xyz[i,0]
                    hex_xyz[p,1] = periodic_xyz[i,1]
                    hex_xyz[p,2] = periodic_xyz[i,2]
                    p = p+1
    return hex_atom_num, a1, a2, b2, c3, hex_xyz
def dump_reader(filename):
    '''
    

    input: dump filename
    output: 
    atom_num : number of atoms
    lenx : length along x-direction
    leny : length along y-direction
    lenz : length along z-direction
    xyz : coordinates

    '''
    f=open(filename, "r")
    lines=f.readlines()
    atom_num = int(lines[3].split('\t')[0])
    lenx = float(lines[5].split(' ')[1])
    leny = float(lines[6].split(' ')[1])
    lenz = float(lines[7].split(' ')[1])
    
    xyz = np.zeros((atom_num,3))
    for i in range (atom_num):
        xyz[i,0] = float(lines[i+9].split(' ')[2])
        xyz[i,1] = float(lines[i+9].split(' ')[3])
        xyz[i,2] = float(lines[i+9].split(' ')[4])
    f.close()
    return atom_num, lenx, leny, lenz, xyz

if __name__=="__main__":
    theta = 1.08
    theta1 = str(theta)
    st = theta1.split('.')
    folder = st[0]+'-'+st[1]
    filename = "simulations/"+folder+"/"+"dump_final.txt";
    
    
    atom_num, lenx, leny, lenz, xyz = dump_reader(filename)
    xyz1= xyz
    filename1 = "POSCAR_"+folder+".txt";
    filename2 = "POSCAR_"+folder+"_hex.txt";
    a1 = lenx
    a2 = 0.0
    b2 = leny
    c3 = lenz
    POSCAR_writer(filename1, atom_num, a1, a2, b2, c3, xyz)
    px = 2
    py = 1
    periodic_atom_num, periodic_lenx, periodic_leny, periodic_lenz, periodic_xyz = periodic_extension(atom_num, lenx, leny, lenz, px, py, xyz)
    tol = 0.005
    hex_atom_num, a1, a2, b2, c3, hex_xyz = \
        recttohex_cutter(atom_num, periodic_atom_num, periodic_lenx, periodic_leny, periodic_lenz, periodic_xyz, tol)
    
    POSCAR_writer(filename2, hex_atom_num, a1, a2, b2, c3, hex_xyz)       


                





