import numpy as np
import pdb

def XYZ_lattice_vectors(fl):

	
	xdata = np.genfromtxt(fl,skip_header=5,max_rows=1)
	ydata = np.genfromtxt(fl,skip_header=6,max_rows=1)

	l_vector1 = np.array([xdata[1]-xdata[0]-abs(xdata[2]),0])
	l_vector2 = np.array([xdata[2],ydata[1]])
	pdb.set_trace()
	print(np.linalg.norm(l_vector1),np.linalg.norm(l_vector2))
	return (l_vector1, l_vector2)

if __name__ == "__main__":
	#print(XYZ_lattice_vectors('../atomic_output_XYZ/atomicrelax_Lm_53.264_nx_41_120.xyz'))
	print(XYZ_lattice_vectors('../atomic_output_XYZ/atomicrelax_Lm_117.156_nx_81_120.xyz'))