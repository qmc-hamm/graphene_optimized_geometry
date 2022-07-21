#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:17:59 2021

@author: tawfiqurrakib
"""
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from operator import itemgetter

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
    identity = np.zeros(atom_num)
    xyz = np.zeros((atom_num,3))
    sigma = np.zeros((atom_num,7))
    energy = np.zeros(atom_num)
    for i in range (atom_num):
        identity[i] = float(lines[i+9].split(' ')[1])
        xyz[i,0] = float(lines[i+9].split(' ')[2])
        xyz[i,1] = float(lines[i+9].split(' ')[3])
        xyz[i,2] = float(lines[i+9].split(' ')[4])
        sigma[i,0] = float(lines[i+9].split(' ')[5])
        sigma[i,1] = float(lines[i+9].split(' ')[6])
        sigma[i,3] = float(lines[i+9].split(' ')[7])
        sigma[i,4] = float(lines[i+9].split(' ')[8])
        sigma[i,5] = float(lines[i+9].split(' ')[9])
        sigma[i,6] = float(lines[i+9].split(' ')[10])
        energy[i] = float(lines[i+9].split(' ')[11])

    f.close()
    return identity, atom_num, lenx, leny, lenz, xyz, sigma, energy


#This code works well fr low twist angle (< 1.47 degrees), 
#because in-plane displacement of atoms are negligible when twist angle is larger

theta = 0.99
theta1 = str(theta)
st = theta1.split('.')
folder = st[0]+'-'+st[1]
filename1 = folder+"/"+"dump_initial.txt";   
filename2 = folder+"/"+"dump_final.txt";  
figname = folder+'/dislocation_width.png'

identity, atom_num, lenx, leny, lenz, xyzi, sigmai, energyi = dump_reader(filename1)
xyzib = xyzi[identity==1,:]  
xyzit = xyzi[identity==2,:] 
sigmaib = sigmai[identity==1,:]  
sigmait = sigmai[identity==2,:] 
identity, atom_num, lenx, leny, lenz, xyzf, sigmaf, energyf = dump_reader(filename2)
xyzfb = xyzf[identity==1,:]  
xyzft = xyzf[identity==2,:] 
sigmafb = sigmaf[identity==1,:]  
sigmaft = sigmaf[identity==2,:] 
dxb = xyzfb[:,0]-xyzib[:,0]
dyb = xyzfb[:,1]-xyzib[:,1]
magb = np.sqrt(dxb**2+dyb**2)
dxt = xyzft[:,0]-xyzit[:,0]
dyt = xyzft[:,1]-xyzit[:,1]
magt = np.sqrt(dxt**2+dyt**2)

width = 1.3
xyzb_1d = xyzib[(xyzib[:,0]>0) & (xyzib[:,0]<width)]
magb_1d = magb[(xyzib[:,0]>0) & (xyzib[:,0]<width)]
arrp = np.vstack((xyzb_1d[:,1],magb_1d))
arrp = np.array(arrp)
arrp = arrp.T
arrp = sorted(arrp, key=itemgetter(0))
arrp = np.array(arrp)
tol = 10

X_Y_Spline = make_interp_spline(arrp[:,0], arrp[:,1])
X = np.linspace(0, leny, 100)
Y = X_Y_Spline(X)



plt.plot(X,Y, '-k', linewidth = 5)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
fig = plt.gcf()
fig.set_size_inches(7.5, 7.5)
plt.tight_layout()
fig.savefig(figname, dpi=200)
plt.show()
