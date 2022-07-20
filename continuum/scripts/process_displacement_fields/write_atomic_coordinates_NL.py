import numpy as np
import scipy as sp
import sympy as sym
from scipy import optimize
from scipy import interpolate
from scipy.interpolate import griddata

import pdb, dill, os, time

import matplotlib.pyplot as plt

def grid_points(X,xgrid,u):

	u1_grid = np.zeros((len(xgrid),2))
	u1_grid[:,0] = griddata((X[:,0], X[:,1]), u[:,0], (xgrid[:,0],xgrid[:,1]), method='linear')#,fill_value=0)
	u1_grid[:,1] = griddata((X[:,0], X[:,1]), u[:,1], (xgrid[:,0],xgrid[:,1]), method='linear')#,fill_value=0)

	return u1_grid

def select_unicell_index(origin,superprimitive1,superprimitive2,lattice):
    """
    origin: length 2 list, 2d original point
    superprimitive: superprimitive vector
    lattice: list of 2d points
    return unitcell indeces
    """
    superprimitive1 =np.array(superprimitive1)
    superprimitive2 =np.array(superprimitive2)
    pt1 =np.array(origin)
    pt2 = pt1+superprimitive1
    pt3 = pt1+superprimitive2
    pt4 = pt1+superprimitive1+superprimitive2
    index = []
    Lat = lattice
    for i in range(len(Lat)):
        polyvector =np.array([superprimitive2,superprimitive1,-superprimitive2,-superprimitive1])
        pointvector =np.array([Lat[i]]*4)-np.array([pt1,pt3,pt4,pt2])
        outp =np.cross(pointvector,polyvector)
        sgn =np.sign(outp) >0
        sgn1 =np.sign(-outp) >0
        if (all(sgn) is True) and (all(sgn1) is False) or (all(sgn) is False) and (all(sgn1) is True):
            index.append(i)

    return index

def unitcell_2d(m,n,size,bond_length=0.142,ang_or = np.pi/3):
    """
    m,n: parameters that define the twisted angle
    bond_length: graphene bond length in nm
    size: integer, the lattice size

    returns 2D unitcell coords, primitive vectors and origin
    """

    # Moon and Koshino's parameter. PRB 85,195458(2012)
    theta = np.arccos((m**2 + n**2 +4*m*n)/(2*(m**2 +n**2 + m*n)))
    rt = lambda t: np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])

    # hexagonal lattice primitive lattice vector
    primitive1 = rt(-theta/2).dot(np.array([np.sqrt(3), 0]))*bond_length
    primitive2 = rt(-theta/2).dot(np.array([np.sqrt(3)/2, 3/2]))*bond_length

    superprimitive1 = m*primitive1+n*primitive2
    superprimitive2 = rt(ang_or)@superprimitive1


    # generate sublattice
    suba = []
    for i in range(3*size):
        for j in range(4*size):
            k = primitive1*(i-2*size+1) + primitive2*(j-1)
            k = k.tolist()
            suba.append(k)
    subb = np.array(suba)+rt(-theta/2).dot(np.array([0,bond_length]))
    subb = subb.tolist()

    # generate lattice
    lattice = suba+subb
    lattice2 = (rt(theta).dot((np.array(lattice).T))).T
    lattice2=lattice2.tolist()

    # select unitcell
    origin=np.array([-0.01,-0.01])
    index = select_unicell_index(origin,superprimitive1,superprimitive2,lattice)
    index2 = select_unicell_index(origin,superprimitive1,superprimitive2,lattice2)

    """
    lattice = np.array(lattice)
    plt.scatter(lattice[:,0],lattice[:,1])
    plt.scatter(lattice[index,0],lattice[index,1])
    plt.axis('equal');plt.show()
    pdb.set_trace()
    """

    unitcell_dn = np.array([lattice[i] for i in index])
    unitcell_up = np.array([lattice2[i] for i in index2])
    return (theta,unitcell_dn,unitcell_up, superprimitive1, superprimitive2,origin)

def generate_twistmoirein_plane_displacements(m,n,nx=None,rho_nx=None,workers=3,load=True,a_cc=1.42):
	#m,n : parameters that determine transformation from graphene to moire lattice see Moon and Koshino's parameter. PRB 85,195458(2012)
	#(future)theta: twist angle in radians
	#rho_nx: linear density of sampling points
	#workers: number of workers to parallelize FFTs over
	
	a0 = 2.46 #Graphene Lattice Dimension Angstroms
	ang_or=2*np.pi/3 #Supercell Angle at origin

	#Generate atomic coordinates
	theta, unitcell_dn, unitcell_up, superprimitive1, superprimitive2, origin = unitcell_2d(m=m,n=n,size=max([m,n]),bond_length=a_cc,ang_or=ang_or)

	Lm = np.linalg.norm(superprimitive1)
	print(r'Twist Angle of %1.3f and Moire Lattice Parameter of %1.3f nm'%(theta*180/np.pi,Lm/10))

	#Define the number of points in a linear direction based on the linear density rho_nx
	if not nx ==None:
		pass
	elif not rho_nx == None:
		nx = int(Lm * rho_nx)
	else:
		print('Define nx or rho_nx')
		return None
	

	disl='screw2d'
	if len(os.getcwd().split('scripts'))>1:
		path = '../../raw/displacement_fields_NL/flat/'
	else:
		path = 'raw/displacement_fields_NL/flat/'
	fl = 'Lm_%1.3f_nx_%d.pkl'%(Lm,nx)

	if __name__ == "__main__":
		# Save Rigid atomic coordinates
		fl_rigid = '../../raw/atomic_output_XYZ/rigid_Lm_%1.3f_nx_%d_%d.xyz'%(Lm,nx,np.round(ang_or*180/np.pi,0))
		save_atomic_coordinates(fl_rigid,unitcell_dn, unitcell_up, superprimitive1, superprimitive2)

	#Check if already have relaxed coordinates
	if os.path.isfile(path+fl) and load:
		print('Loading previous relaxed displacement field')
		with open(path+fl, 'rb') as f:
			cell = dill.load(f)
	else:
		print('Please Calculate a Displacement Field')
		pdb.set_trace()
		return None



	#Relax atomic coordinates with displacement field


	#Extract local displacement fields from calculation object
	u1_t = cell.u1().reshape((2,np.prod(cell.X.shape[1:]))).T
	u2_t = cell.u2().reshape((2,np.prod(cell.X.shape[1:]))).T

	#Extract associated positions from calculation object
	X = cell.X.reshape((2,np.prod(cell.X.shape[1:]))).T

	#Tile Continuum field in space to ensure that atomic coordinates are within the boundaries
	X_shift = X + np.array([0,Lm])
	X_shiftl = X - np.array([Lm*3**0.5,0])

	X_extended = np.concatenate([X,X_shift,X_shiftl])
	u1_t_extended = np.concatenate([u1_t,u1_t,u1_t])
	u2_t_extended = np.concatenate([u2_t,u2_t,u2_t])

	#Shift continuum positions such that (0,0) is AA stacking
	X_continuum = X_extended - np.array([Lm*3**0.5/3, 0])

	#Interpolate continuum field to the atomic coordinates
	U_displacements_bottom = grid_points(X_continuum,unitcell_dn,u1_t_extended)
	U_displacements_top = grid_points(X_continuum,unitcell_up,u2_t_extended)

	#pdb.set_trace()
	#Update atomic coordinates with the displacement fields
	atomic_coordinates_displaced_top = unitcell_up + U_displacements_top
	atomic_coordinates_displaced_bottom = unitcell_dn + U_displacements_bottom
	
	#change from 2d to 3d space
	atomic_coordinates_displaced_top = np.concatenate([atomic_coordinates_displaced_top,cell.d0*np.ones((len(atomic_coordinates_displaced_top),1))],axis=1)
	atomic_coordinates_displaced_bottom = np.concatenate([atomic_coordinates_displaced_bottom,np.zeros((len(atomic_coordinates_displaced_bottom),1))],axis=1)
	superprimitive1 = np.concatenate([superprimitive1,[0]])
	superprimitive2 = np.concatenate([superprimitive2,[0]])

	if __name__ == "__main__":
		fl_relax = '../../raw/atomic_output_XYZ/atomicrelax_Lm_%1.3f_nx_%d_%d.xyz'%(Lm,nx,np.round(ang_or*180/np.pi,0))
		save_atomic_coordinates(fl_relax,atomic_coordinates_displaced_bottom, atomic_coordinates_displaced_top, superprimitive1, superprimitive2)

	return (atomic_coordinates_displaced_bottom, atomic_coordinates_displaced_top, superprimitive1, superprimitive2)

def save_atomic_coordinates(filename,atomic_bottom,atomic_top,superprimitive1,superprimitive2,d=3.39):
	tags = ['C','Si']

	#Rotate atoms so that superprimitive1 is parallel to x-axis
	theta_sp1 = np.arctan(superprimitive1[1]/superprimitive1[0])
	rt = lambda t: np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])

	superprimitive1_rot = rt(-theta_sp1).dot(superprimitive1)
	superprimitive2_rot = rt(-theta_sp1).dot(superprimitive2)

	atomic_bottom_rot = rt(-theta_sp1).dot(atomic_bottom.T).T
	atomic_top_rot = rt(-theta_sp1).dot(atomic_top.T).T

	fl = open(filename, 'w')
	#pdb.set_trace()
	natoms = len(atomic_bottom)+len(atomic_top)
	
	fl.write('ITEM: TIMESTEP\n%d\n'%(0))
	fl.write('ITEM: NUMBER OF ATOMS\n%d\n'%(natoms))
	fl.write('ITEM: BOX BOUNDS xy xz yz pp pp pp \n')
	fl.write('%1.8f %1.8f %1.8f \n'%(superprimitive2_rot[0]*(superprimitive2_rot[0]<0),superprimitive1_rot[0]+superprimitive2_rot[0]*(superprimitive2_rot[0]>0),superprimitive2_rot[0]))
	fl.write('%1.8f %1.8f %1.8f \n'%(0,superprimitive2_rot[1],0))
	fl.write('%1.3f %1.3f %1.3f \n'%(-20,20,0))
	fl.write('ITEM: ATOMS type x y z\n')

	for j in range(len(atomic_bottom_rot)):
		line = (tags[0] + ' ' + str(atomic_bottom_rot[j,0]) + ' ' + str(atomic_bottom_rot[j,1]) + ' ' + str(0) + '\n')
		fl.write(line)

	for j in range(len(atomic_top_rot)):
		line = (tags[1] + ' ' + str(atomic_top_rot[j,0]) + ' ' + str(atomic_top_rot[j,1]) + ' ' + str(d) + '\n')
		fl.write(line)
	

os.getcwd()
if __name__ == "__main__":
	generate_twistmoirein_plane_displacements(12,13, nx = 41,rho_nx=0.77)
	#generate_twistmoirein_plane_displacements(27,28, 0.692)
	#generate_twistmoirein_plane_displacements(31,32, 0.31)
	#XYZ_lattice_vectors('../atomic_output_XYZ/atomicrelax_Lm_53.264_nx_41.xyz')