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
    cbar.set_label('In-plane displacement ($\AA$)',rotation=90, fontsize = 20, fontweight = 'bold', labelpad=20)
    cbar.ax.tick_params(labelsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    fig = plt.gcf()
    fig.set_size_inches(7, 10)
    plt.tight_layout()
    fig.savefig(figname, dpi=200)
    plt.show()

def scatterplotter(x,y,z, dia_dot, vmin, vmax, caption, filename):
    plt.figure()
    plt.scatter( x, y, c=z, cmap = 'jet', s=dia_dot, alpha=0.8)
    cbar = plt.colorbar()
    cbar.set_label(caption,rotation=90, fontsize = 20, fontweight = 'bold', labelpad=20)
    cbar.ax.tick_params(labelsize = 18)
    plt.clim(vmin = vmin, vmax = vmax)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    fig = plt.gcf()
    fig.set_size_inches(7, 10)
    plt.tight_layout()
    fig.savefig(filename, dpi=200)


theta = '0.99'
theta1 = str(theta)
st = theta1.split('.')
folder = st[0]+'-'+st[1]
filename1 = "dump_initial.txt";   
filename2 = "dump_final.txt";  
figname1 = 'in-plane_bottom.png'
figname2 = 'in-plane_top.png'
identity, atom_num, lenx, leny, lenz, xyzi, sigmai, energyi = dump_reader(filename1)
xyzib = xyzi[identity==1,:]  
xyzit = xyzi[identity==2,:] 
sigmai = sigmai/10
sigmai = sigmai*3.4/lenz
sigmaib = sigmai[identity==1,:]  
sigmait = sigmai[identity==2,:] 
identity, atom_num, lenx, leny, lenz, xyzf, sigmaf, energyf = dump_reader(filename2)
xyzfb = xyzf[identity==1,:]  
xyzft = xyzf[identity==2,:] 
sigmaf = sigmaf/10
sigmaf = sigmaf*lenz/3.4
sigmafb = sigmaf[identity==1,:]  
sigmaft = sigmaf[identity==2,:] 
dxb = xyzfb[:,0]-xyzib[:,0]
dyb = xyzfb[:,1]-xyzib[:,1]
magb = np.sqrt(dxb**2+dyb**2)
dxt = xyzft[:,0]-xyzit[:,0]
dyt = xyzft[:,1]-xyzit[:,1]
magt = np.sqrt(dxt**2+dyt**2)
quiverplotter(xyzib, dxb, dyb, magb, figname1)
quiverplotter(xyzit, dxt, dyt, magt, figname2)

scatterplotter(xyzfb[:,0], xyzfb[:,1], xyzfb[:,2], 5, 7.0, 8.0, "z-displacement ($\AA$)","out-disp-bottom.png")
scatterplotter(xyzft[:,0], xyzft[:,1], xyzft[:,2], 5, 10.4, 11.3, "z-displacement ($\AA$)","out-disp-top.png")
energyfb = energyf[identity==1]  
energyft = energyf[identity==2] 
scatterplotter(xyzfb[:,0], xyzfb[:,1], energyfb[:], 5, np.min(energyfb), np.max(energyfb), "Energy (eV/atom)","energy-bottom.png")
scatterplotter(xyzft[:,0], xyzft[:,1], energyft[:], 5, np.min(energyft), np.max(energyft), "Energy (eV/atom)","energy-top.png")
scatterplotter(xyzfb[:,0], xyzfb[:,1], sigmafb[:,0], 5, np.min(sigmafb[:,0]), np.max(sigmafb[:,0]), "$\sigma_{xx}$ (MPa)","sigmaxx-bottom.png")
scatterplotter(xyzft[:,0], xyzft[:,1], sigmaft[:,0], 5, np.min(sigmaft[:,0]), np.max(sigmaft[:,0]), "$\sigma_{xx}$ (MPa)","sigmaxx-top.png")
scatterplotter(xyzfb[:,0], xyzfb[:,1], sigmafb[:,1], 5, np.min(sigmafb[:,1]), np.max(sigmafb[:,1]), "$\sigma_{yy}$ (MPa)","sigmayy-bottom.png")
scatterplotter(xyzft[:,0], xyzft[:,1], sigmaft[:,1], 5, np.min(sigmaft[:,1]), np.max(sigmaft[:,1]), "$\sigma_{yy}$ (MPa)","sigmayy-top.png")


