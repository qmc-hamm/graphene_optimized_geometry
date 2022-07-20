# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:54:56 2021

@author: trakib2
"""

import numpy as np 
import h5py
from ase import Atoms
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt



with h5py.File('best_estimate.hdf5', 'r') as hf:
    L = list(hf.keys())
    ID = L[0]
    identity = hf.get(ID)
    lattice_vector = identity.get('lattice_vector')
    lattice_vector = np.array(lattice_vector)
    xyz = identity.get('xyz')  #coordinates
    xyz = np.array(xyz)
    m = identity.get('m')
    m = int(np.array(m))
    n = identity.get('n')
    n = int(np.array(n))
    theta = identity.get('theta')  #twist angles
    theta = float(np.array(theta))
    method= identity.get('method')
    date = identity.get('date')

natoms = len(xyz)
st = 'C'+str(natoms)
atoms = Atoms(st, cell=lattice_vector)
atoms.set_positions(xyz)
fig, ax = plt.subplots()
plot_atoms(atoms, ax, radii=0.3)
#date1 = identity.get('date')
#date = date1.attrs["2"]
#met = identity.get('method')
#method = met.attrs["1"]



