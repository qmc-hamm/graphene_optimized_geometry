#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:47:38 2021

@author: tawfiqurrakib
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:17:59 2021

@author: tawfiqurrakib
"""
import numpy as np 
import matplotlib.pyplot as plt

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
def quiverplotter(xyzi, dx, dy, mag, figname):
    plt.figure()
    plt.quiver(xyzi[:,0],xyzi[:,1],dx,dy, mag, cmap = 'jet', headlength = 20 , headwidth = 24, headaxislength = 9) 
    cbar = plt.colorbar()
    cbar.set_label('difference of in-plane displacement ($\AA$)',rotation=90, fontsize = 20, fontweight = 'bold', labelpad=20)
    cbar.ax.tick_params(labelsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    fig = plt.gcf()
    fig.set_size_inches(7.5, 10)
    plt.tight_layout()
    fig.savefig(figname, dpi=200)
    plt.show()



theta = '0.99'
theta1 = str(theta)
st = theta1.split('.')
folder = st[0]+'-'+st[1]
folder1 = "KC-REBO/raw/simulations/"+folder+"/"
filename1 = folder1+"/"+"dump_initial.txt";   
filename2 = folder1+"/"+"dump_final.txt";  
folder2 = "FitttedKC-REBO/raw/simulations/"+folder+"/"
filename3 = folder2+"/"+"dump_initial.txt";   
filename4 = folder2+"/"+"dump_final.txt";  
figname1 = folder1+'/diff_in-plane_bottom.png'
figname2 = folder1+'/diff_in-plane_top.png'
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
identityq, atom_num, lenx, leny, lenz, xyziq, sigmaiq, energyiq = dump_reader(filename3)
xyzqib = xyziq[identityq==1,:]  
xyzqit = xyziq[identityq==2,:] 
identityq, atom_num, lenx, leny, lenz, xyzfq, sigmafq, energyfq = dump_reader(filename4)
xyzqfb = xyzfq[identityq==1,:]  
xyzqft = xyzfq[identityq==2,:] 
dxb = xyzfb[:,0]-xyzib[:,0]
dyb = xyzfb[:,1]-xyzib[:,1]
magb = np.sqrt(dxb**2+dyb**2)
dxt = xyzft[:,0]-xyzit[:,0]
dyt = xyzft[:,1]-xyzit[:,1]
magt = np.sqrt(dxt**2+dyt**2)
dxbq = xyzqfb[:,0]-xyzib[:,0]
dybq = xyzqfb[:,1]-xyzib[:,1]
magbq = np.sqrt(dxbq**2+dybq**2)
dxtq = xyzqft[:,0]-xyzqit[:,0]
dytq = xyzqft[:,1]-xyzqit[:,1]
magtq = np.sqrt(dxtq**2+dytq**2)

DXb = dxbq-dxb
DXt = dxtq-dxt
DYb = dybq-dyb
DYt = dytq-dyt
MAGb = magbq-magb
MAGt = magtq-magt
quiverplotter(xyzib, DXb, DYb, MAGb, figname1)
quiverplotter(xyzit, DXt, DYt, MAGt, figname2)
