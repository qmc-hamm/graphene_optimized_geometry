Twisted Bilayer Graphene -- Geometry Library
======================================================================
  
This repository contains a compilation of relaxed geometries for twisted bilayer graphene for a variety of twist angles, as obtained using a selection of different methods: first-principles density functional theory, classical interatomic potentials, and linear continuum elasticity theory.  There is also a directory containing rigidly twisted bilayers, without relaxation. 

The purpose of this repository is to enable sharing of geometries and facilitate comparisons of geometries obtained using different methods. These geometries can, for example, be analyzed using tight-binding models, etc. 

Directories: 
--------------

Each directory contains geometries obtained using select methods. Each sub-directory contains an hdf5 file containing the geometries.  

- rigid: unrelaxed twisted bilayer graphene (i.e. rigid rotations without symmetry breaking) 
- continuum: linear elasticity theory, based on [Phys. Rev. B 102, 184107 (2020)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.184107). 
- interatomic_potentials: empirical potential combinations including (i) Rebo + Kolmogorov-Crespi (KC-Ouyang), (ii) Rebo + QMC-fitted KC (KC-Mick), (iii) Rebo + DFT-D2 fitted KC, and (iv) Rebo + DFT-D3 fitted KC  
- DFT: PBE+TSvdW relaxed 


To Use:
--------------

- best_estimate_reader.py : reads the hdf5 file and stores the following <br>
  - L: list of UUIDs for each geometry  <br>
  - lattice_vector: set of lattice vectors  <br>
  - xyz: xyz coordinates  <br>
  - m,n : indices of twisted bilayer graphene according to [Phys Rev B 90, 155451 (2014)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.155451)  <br>
  - theta : twist angle  <br>
  - method : how the geometries are obtained  <br>
  - date : date created 


  - do we want one script in each directory, or just one parent script in main directory?  They are carbon copies of each other. 
    
- create_best_estimate.py : add new data set to hdf5 file 
  - do we want this open/available in general? not sure. 

- mn.txt : catalog of available (m,n) values for available twist angles 






