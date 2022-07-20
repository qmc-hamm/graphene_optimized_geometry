import numpy as np
import scipy as sp
import sympy as sym
from scipy import optimize
from scipy import interpolate
from scipy.interpolate import griddata

import pdb, dill, os, time

import matplotlib.pyplot as plt

class flat_displacement_field():
	#find the displacement field for supercells constrained to be flat
	def __init__(self,eps,L,n,workers=1):
		"""
		eps: the strain in each layer given as a list
		L: the supercell dimensions given as a list
		n: the number of fourier components to solve for
		"""
		#Define elasic constants of each layer
		self.createCijkl()
		self.L = L
		self.n = n

		#Define Material Constants -- See 3D GSFE from Dai and Srolovitz
		self.a0 = 2.46#*1e-10
		self.d0 = 3.39#*1e-10
		self.Cnn0 = 38.*1e12*6.24e-15 #GPa to mJ/m^3 Convert from mJ/m^3 to eV/A^3
		self.gamma_coh = 308.9*6.24e-5 #Convert from mJ/m^2 to eV/A^2#sym.Symbol('gamma_coh')
		self.k0 = self.Cnn0/self.d0
		self.kappa = 22.08e-20*6.242e18#sym.Symbol('kappa')

		self.eps1_0 = eps[0]
		self.eps2_0 = eps[1]

		#Define real space mesh
		x = np.linspace(0,L[0],n[0],endpoint=False)
		y = np.linspace(0,L[1],n[1],endpoint=False)
		xx, yy = np.meshgrid(x,y)

		self.X = np.array([xx,yy])
		self.dX = np.array([x[1]-x[0],y[1]-y[0]])
		self.dA = self.dX[0]*self.dX[1]

		#Define reciprocal space mesh
		Kx = 2*np.pi*sp.fft.rfftfreq(self.n[0],self.dX[0])
		Ky = 2*np.pi*sp.fft.fftfreq(self.n[1],self.dX[1])

		Kxx, Kyy = np.meshgrid(Kx,Ky)
		self.K = np.array([Kxx,Kyy])

		# Initialize u1, u2, f1, f2
		self.u1_0 = np.array([self.eps1_0[0,0]*self.X[0]+self.eps1_0[0,1]*self.X[1],self.eps1_0[1,0]*self.X[0]+self.eps1_0[1,1]*self.X[1]])
		self.u2_0 = np.array([self.eps2_0[0,0]*self.X[0]+self.eps2_0[0,1]*self.X[1],self.eps2_0[1,0]*self.X[0]+self.eps2_0[1,1]*self.X[1]])

		#define how many workers can do the FFTs in parallel
		self.workers = workers

		#Load/Calculate 3D GSFE
		dill.settings['recurse'] = True

		#pdb.set_trace()
		load = True
		location_functions = 'interlayer_energy_functions/'
		if os.path.isfile(location_functions + "lam_dGamma_du") and load:
			self.lam_Gamma=dill.load(open(location_functions + "lam_Gamma", "rb"))
			self.lam_dGamma_du=dill.load(open(location_functions + "lam_dGamma_du", "rb"))
			self.lam_d2Gamma_du2=dill.load(open(location_functions + "lam_d2Gamma_du2", "rb"))
			self.lam_dGamma_df=dill.load(open(location_functions + "lam_dGamma_df", "rb"))
			self.lam_dGamma_dfp=dill.load(open(location_functions + "lam_dGamma_dfp", "rb"))
			self.lam_dGamma_dfm=dill.load(open(location_functions + "lam_dGamma_dfm", "rb"))
		else:
			print('Creating Interlayer Energy Functions')
			#Define Functions
			phi = sym.Symbol('phi')
			psi = sym.Symbol('psi')

			f_p = sym.Symbol('fp')
			f_m = sym.Symbol('fm')

			C_gam = np.array([21.336,-6.127,-1.128,0.143,np.sqrt(3)*-6.127,-np.sqrt(3)*0.143])*6.24e-5 #Convert from mJ/m^2 to eV/A^2# as given in Dai 3D GSFE[21.336,-6.127,-1.128,0.143,np.sqrt(3)*-6.127,-np.sqrt(3)*0.143]
			gamma = F(phi,psi,C_gam,self.a0)

			C_d = np.array([3.47889,-0.02648,-0.00352,0.00037,np.sqrt(3)*-0.02648,-np.sqrt(3)*0.00037]) # as given in Dai 3D GSFE
			d = F(phi,psi,C_d,self.a0)

			falpha = alpha(d,gamma,self.k0,self.gamma_coh)

			fA = A(falpha,gamma,d,self.gamma_coh)
			fB = B(falpha,gamma,d,self.gamma_coh)

			f_perp = f_p - f_m + self.d0
			Gamma = -fB*(d/f_perp)**4 + self.gamma_coh + fA*sym.exp(-falpha*f_perp)#
			lam_Gamma = sym.lambdify([phi,psi,f_p,f_m],Gamma)

			dGamma_du = sym.derive_by_array(Gamma,[phi,psi])
			lam_dGamma_du = sym.lambdify([phi,psi,f_p,f_m],dGamma_du)

			d2Gamma_du2 = sym.derive_by_array(dGamma_du,[phi,psi])
			lam_d2Gamma_du2 = sym.lambdify([phi,psi,f_p,f_m],d2Gamma_du2)

			dGamma_dfp = sym.diff(Gamma,f_p)
			lam_dGamma_dfp = sym.lambdify([phi,psi,f_p,f_m],dGamma_dfp)

			dGamma_dfm = sym.diff(Gamma,f_m)
			lam_dGamma_dfm = sym.lambdify([phi,psi,f_p,f_m],dGamma_dfm)

			f_perp = sym.Symbol('f_perp')

			Gamma = fA*sym.exp(-falpha*f_perp)-fB*(d/f_perp)**4 + self.gamma_coh
			dGamma_df = sym.diff(Gamma,f_perp)
			lam_dGamma_df = sym.lambdify([phi,psi,f_perp],dGamma_df)

			#Save Functions
			dill.dump(lam_Gamma, open(location_functions + "lam_Gamma", "wb"))
			dill.dump(lam_dGamma_du, open(location_functions + "lam_dGamma_du", "wb"))
			dill.dump(lam_d2Gamma_du2, open(location_functions + "lam_d2Gamma_du2", "wb"))
			dill.dump(lam_dGamma_dfp, open(location_functions + "lam_dGamma_dfp", "wb"))
			dill.dump(lam_dGamma_dfm, open(location_functions + "lam_dGamma_dfm", "wb"))
			dill.dump(lam_dGamma_df, open(location_functions + "lam_dGamma_df", "wb"))

	def createCijkl(self,E=18.5,mu = 5.49):
		"""
		E = 16.93#*100
		G = 6.9

		lam = G*(E-2*G)/(3*G-E)

		C11 = lam+2*G
		C12 = lam
		C66 = G
		"""
		#Rebo Single layer - Research Update 6/18
		#C12 = 7.15
		C11 = E
		C66 = mu#5.49
		#C11 = C12+2*C66
		C12 = C11-2*C66

		C13 = 0#0.026
		C33 = 0#0.64
		C44 = 0#.011 #Check

		C14 = 0#0.002
		C15 = 0#-0.002

		#C34;C35;C36;C16;C26;C45 are all symmetrically zero
		C34 = 0#.08

		Cijkl = np.zeros((3,3,3,3))
		#C11;C22
		Cijkl[0,0,0,0], Cijkl[1,1,1,1] = C11*np.ones(2)
		#C12
		Cijkl[0,0,1,1], Cijkl[1,1,0,0] = C12*np.ones(2)
		#C66
		Cijkl[0,1,0,1], Cijkl[1,0,1,0], Cijkl[0,1,1,0], Cijkl[1,0,0,1] = C66*np.ones(4)
		#C33
		Cijkl[2,2,2,2] =  C33
		#C13; C23
		Cijkl[2,2,1,1], Cijkl[2,2,0,0],	Cijkl[0,0,2,2], Cijkl[1,1,2,2] = C13*np.ones(4)
		#C44
		Cijkl[2,1,2,1], Cijkl[1,2,1,2], Cijkl[2,1,1,2], Cijkl[1,2,2,1] = C44*np.ones(4)
		#C55
		Cijkl[0,2,0,2], Cijkl[2,0,2,0], Cijkl[0,2,2,0], Cijkl[2,0,0,2] = C44*np.ones(4)
		#C14
		Cijkl[2,1,0,0], Cijkl[1,2,0,0], Cijkl[0,0,2,1], Cijkl[0,0,1,2] = C14*np.ones(4)
		#C24
		Cijkl[2,1,1,1], Cijkl[1,2,1,1], Cijkl[1,1,2,1], Cijkl[1,1,1,2] = -C14*np.ones(4)
		#C56 - 1312
		Cijkl[0,2,0,1], Cijkl[0,2,1,0], Cijkl[2,0,0,1], Cijkl[2,0,1,0] = C14*np.ones(4)
		Cijkl[0,1,0,2], Cijkl[1,0,0,2], Cijkl[0,1,2,0], Cijkl[1,0,2,0] = C14*np.ones(4)

		#C15
		Cijkl[2,0,0,0], Cijkl[0,2,0,0], Cijkl[0,0,2,0], Cijkl[0,0,0,2] = C15*np.ones(4)
		#C25
		Cijkl[2,0,1,1], Cijkl[0,2,1,1], Cijkl[1,1,2,0], Cijkl[1,1,0,2] = -C15*np.ones(4)
		#C46
		Cijkl[1,2,0,1], Cijkl[2,1,0,1], Cijkl[1,2,1,0], Cijkl[2,1,1,0] = -C15*np.ones(4)
		Cijkl[0,1,1,2], Cijkl[0,1,2,1], Cijkl[1,0,1,2], Cijkl[1,0,2,1] = -C15*np.ones(4)

		#C34
		#Cijkl[1,2,2,2], Cijkl[2,1,2,2], Cijkl[2,2,1,2], Cijkl[2,2,2,1] = C34*np.ones(4)
		#C35
		Cijkl[0,2,2,2], Cijkl[2,0,2,2], Cijkl[2,2,0,2], Cijkl[2,2,2,0] = C34*np.ones(4)

		self.Cijkl = Cijkl[:2,:2,:2,:2]

	def real_to_complex(self,z):      # real vector of length 2n -> complex of length n
	    #pdb.set_trace()
	    return z[:len(z)//2] + 1j * z[len(z)//2:]

	def complex_to_real(self,z):      # complex vector of length n -> real of length 2n
	    #pdb.set_trace()
	    return np.concatenate((np.real(z), np.imag(z)))

	def u1(self):
		return np.real(sp.fft.irfft2(self.u1p_t,s=self.n[::-1],workers=self.workers))

	def u2(self):
		return np.real(sp.fft.irfft2(self.u2p_t,s=self.n[::-1],workers=self.workers))

	def f1(self):
		return np.real(sp.fft.irfft2(self.f1p_t,s=self.n[::-1],workers=self.workers))

	def f2(self):
		return np.real(sp.fft.irfft2(self.f2p_t,s=self.n[::-1],workers=self.workers))

	def E_srol_BLGr(self,d1,d2,minimize='du'):
		if minimize=='du':
			up = self.real_to_complex(d1)
			fp = self.real_to_complex(d2)
		elif minimize == 'df':
			up = self.real_to_complex(d2)
			fp = self.real_to_complex(d1)

		self.u1p_t, self.u2p_t = up.reshape((2,2,self.K.shape[1],self.K.shape[2]))
		self.f1p_t, self.f2p_t = fp.reshape((2,self.K.shape[1],self.K.shape[2]))

		#pdb.set_trace()
		df1_dx = (np.real(sp.fft.irfft2(np.einsum('ikl,kl->ikl',1j*self.K,self.f1p_t),s=self.n[::-1],workers=self.workers)))#.T*np.sqrt(n[0]*n[1])).T
		df2_dx = (np.real(sp.fft.irfft2(np.einsum('ikl,kl->ikl',1j*self.K,self.f2p_t),s=self.n[::-1],workers=self.workers)))

		d2f1_dx2 = (np.real(sp.fft.irfft2(np.einsum('ikl,jkl,kl->ijkl',1j*self.K,1j*self.K,self.f1p_t),s=self.n[::-1],workers=self.workers)))#.T*n*n).T
		d2f2_dx2 = (np.real(sp.fft.irfft2(np.einsum('ikl,jkl,kl->ijkl',1j*self.K,1j*self.K,self.f2p_t),s=self.n[::-1],workers=self.workers)))


		F1 = np.einsum('ikl,jkl->ijkl',df1_dx,df1_dx)/2
		F2 = np.einsum('ikl,jkl->ijkl',df2_dx,df2_dx)/2

		H1 = d2f1_dx2[0,0] + d2f1_dx2[1,1]
		H2 = d2f2_dx2[0,0] + d2f2_dx2[1,1]

		self.fperp = self.f2()-self.f1()+self.d0

		#Numeric Derivatives in Fourier Space
		du1_dx = np.real(sp.fft.irfft2(np.einsum('ikl,jkl->ijkl',1j*self.K,self.u1p_t),s=self.n[::-1],workers=self.workers))
		du2_dx = np.real(sp.fft.irfft2(np.einsum('ikl,jkl->ijkl',1j*self.K,self.u2p_t),s=self.n[::-1],workers=self.workers))

		eps_1 = (np.einsum('ij,kl',self.eps1_0,np.ones(du1_dx.shape[2:]))+(du1_dx+np.transpose(du1_dx,(1,0,2,3)))/2+F1)
		eps_2 = (np.einsum('ij,kl',self.eps2_0,np.ones(du1_dx.shape[2:]))+(du2_dx+np.transpose(du2_dx,(1,0,2,3)))/2+F2)

		self.uperp = ((self.u2_0-self.u1_0) + (self.u2()-self.u1()) + np.einsum('ijk,jk->ijk',(df2_dx+df1_dx)/2,self.fperp)) #Might need to add initial displacement u0

		self.E_x = np.real(self.lam_Gamma(self.uperp[0],self.uperp[1],self.f2(),self.f1()))

		self.Ee_x1 = np.real(np.einsum('ijkl,ijmn,klmn->mn',self.Cijkl,eps_1,eps_1)/2)
		self.Ee_x2 = np.real(np.einsum('ijkl,ijmn,klmn->mn',self.Cijkl,eps_2,eps_2)/2)

		self.Eb_1 = self.kappa*H1**2/2
		self.Eb_2 = self.kappa*H2**2/2

		self.E = (np.sum(self.Ee_x1) + np.sum(self.Ee_x2) + np.sum(self.Eb_1) + np.sum(self.Eb_2) + np.sum(self.E_x))*self.dA
		#pdb.set_trace()
		return self.E

	def dE_du_srol_BLGr(self,up,fp):
		up = self.real_to_complex(up)
		fp = self.real_to_complex(fp)

		self.u1p_t, self.u2p_t = up.reshape((2,2,self.K.shape[1],self.K.shape[2]))
		self.f1p_t, self.f2p_t = fp.reshape((2,self.K.shape[1],self.K.shape[2]))

		df1_dx = np.real(sp.fft.irfft2(np.einsum('ikl,kl->ikl',1j*self.K,self.f1p_t),s=self.n[::-1],workers=self.workers))
		df2_dx = np.real(sp.fft.irfft2(np.einsum('ikl,kl->ikl',1j*self.K,self.f2p_t),s=self.n[::-1],workers=self.workers))

		F1 = np.einsum('ikl,jkl->ijkl',df1_dx,df1_dx)/2
		F2 = np.einsum('ikl,jkl->ijkl',df2_dx,df2_dx)/2
		Fp1 = sp.fft.rfft2(F1,workers=self.workers)#[:,:,:self.n[1]//2+1,:self.n[0]//2+1]
		Fp2 = sp.fft.rfft2(F2,workers=self.workers)#[:,:,:self.n[1]//2+1,:self.n[0]//2+1]

		self.fperp = self.f2()-self.f1()+self.d0

		#Numeric Derivatives in Fourier Space
		du1_dx = np.real(sp.fft.irfft2(np.einsum('ikl,jkl->ijkl',1j*self.K,self.u1p_t),s=self.n[::-1],workers=self.workers))
		du2_dx = np.real(sp.fft.irfft2(np.einsum('ikl,jkl->ijkl',1j*self.K,self.u2p_t),s=self.n[::-1],workers=self.workers))

		d2u1_dx2 = np.real(sp.fft.irfft2(np.einsum('ikl,jkl,mkl->ijmkl',1j*self.K,1j*self.K,self.u1p_t),s=self.n[::-1],workers=self.workers))
		d2u2_dx2 = np.real(sp.fft.irfft2(np.einsum('ikl,jkl,mkl->ijmkl',1j*self.K,1j*self.K,self.u2p_t),s=self.n[::-1],workers=self.workers))

		self.uperp = ((self.u2_0-self.u1_0) + (self.u2()-self.u1()) + np.einsum('ijk,jk->ijk',(df2_dx+df1_dx),self.fperp/2)) #Might need to add initial displacement u0

		dG_du = self.lam_dGamma_du(self.uperp[0],self.uperp[1],self.f2(),self.f1())
		dGamp = sp.fft.rfft2(dG_du,workers=self.workers)#[:,:self.n[1]//2+1,:self.n[0]//2+1]

		CKF1 = 1j*np.einsum('ijkl,jmn,klmn->imn',self.Cijkl,self.K,Fp1)
		CKF2 = 1j*np.einsum('ijkl,jmn,klmn->imn',self.Cijkl,self.K,Fp2)

		dEeu_du1 = -sp.fft.rfft2(np.einsum('ijkl,jklmn->imn',self.Cijkl,d2u1_dx2+np.transpose(d2u1_dx2,(0,2,1,3,4)))/2,workers=self.workers)#[:,:self.n[1]//2+1,:self.n[0]//2+1]
		dEeu_du2 = -sp.fft.rfft2(np.einsum('ijkl,jklmn->imn',self.Cijkl,d2u2_dx2+np.transpose(d2u2_dx2,(0,2,1,3,4)))/2,workers=self.workers)#[:,:self.n[1]//2+1,:self.n[0]//2+1]
		#pdb.set_trace()
		dE_du1 =   - dGamp + dEeu_du1 - CKF1#
		dE_du2 =   + dGamp + dEeu_du2 - CKF2#
		#dE_du1.real=0#/=2
		#dE_du2.real=0#/=2
		dE_du = np.array([dE_du1,dE_du2]).flatten()*self.dA/np.prod(self.n)

		return self.complex_to_real(dE_du)

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

def unitcell_2d(m,n,size,bond_length=1.42):
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
    superprimitive2 = rt(np.pi*1/3)@superprimitive1


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

def generate_displacement_field(m,n, rho_nx,workers=3,load=True,a_cc=1.42):
	#m,n : parameters that determine transformation from graphene to moire lattice see Moon and Koshino's parameter. PRB 85,195458(2012)
	#(future)theta: twist angle in radians
	#rho_nx: linear density of sampling points
	#workers: number of workers to parallelize FFTs over
	a0 = 2.46

	#Generate atomic coordinates
	theta, unitcell_dn, unitcell_up, superprimitive1, superprimitive2, origin = unitcell_2d(m=m,n=n,size=max([m,n]),bond_length=a_cc)

	Lm = np.linalg.norm(superprimitive1)
	print(r'Twist Angle of %1.3f and Moire Lattice Parameter of %1.3f nm'%(theta*180/np.pi,Lm/10))

	#Define the number of points in a linear direction based on the linear density rho_nx
	nx = int(Lm * rho_nx)
	#pdb.set_trace()
	disl='screw2d'
	path = '../../raw/displacement_fields_NL/flat/'
	fl = 'Lm_%1.3f_nx_%d.pkl'%(Lm,nx)

	# Save Rigid atomic coordinates
	#fl_rigid = '../atomic_output_XYZ/rigid_Lm_%1.3f_nx_%d.xyz'%(Lm,nx)
	#save_atomic_coordinates(fl_rigid,unitcell_dn, unitcell_up, superprimitive1, superprimitive2)

	#Check if already have relaxed coordinates
	if os.path.isfile(path+fl) and load:
		print('Displacement field already exists')
		with open(path+fl, 'rb') as f:
			cell = dill.load(f)
	else:
		#Define input parameters to displacement field minimization
		L = [Lm*3**0.5,Lm]
		n = [int(nx*3**0.5),nx]

		eps1_0 = np.zeros((2,2))
		eps1_0[0,1] = a0/L[1]/2
		eps1_0[1,0] = -a0/L[1]/2
		eps2_0 = np.zeros((2,2))
		eps2_0[0,1] = -a0/L[1]/2
		eps2_0[1,0] = a0/L[1]/2

		eps = [eps1_0,eps2_0]

		#Define Initial Guess for real FFT
		u1p = np.zeros(np.array([2,n[1],n[0]//2+1]),dtype='complex128')
		u2p = np.zeros(np.array([2,n[1],n[0]//2+1]),dtype='complex128')

		f1p = np.zeros(np.array([n[1],n[0]//2+1]),dtype='complex128')
		f2p = np.zeros(np.array([n[1],n[0]//2+1]),dtype='complex128')

		up0r = np.array([u1p,u2p]).flatten()
		fp0r = np.array([f1p,f2p]).flatten()

		#Initialize Continuum Cell
		cell = flat_displacement_field(eps,L,n,workers)

		#Minimize the in-plane displacement components
		res = sp.optimize.minimize(cell.E_srol_BLGr,cell.complex_to_real(up0r),args=(cell.complex_to_real(fp0r)),jac=cell.dE_du_srol_BLGr)

		print('Minimization Successful')
		print(nx,res.success,res.nit,res.nfev,res.njev)

		if not os.path.isdir(path):
			os.makedirs(path)

		#Save Relaxed Displacement field if minimization successful
		if res.success:
			with open(path + fl, 'wb') as f:
				dill.dump(cell,f)
		else:
			with open(path+'unsuccessful'+fl, 'wb') as f:
				dill.dump(cell,f)



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

	#Update atomic coordinates with the displacement fields
	atomic_coordinates_displaced_top = unitcell_up + U_displacements_top #Convert displacement field to nanometers
	atomic_coordinates_displaced_bottom = unitcell_dn + U_displacements_bottom #Convert displacement field to nanometers

	fl_relax = '../../raw/atomic_output_XYZ/atomicrelax_Lm_%1.3f_nx_%d.xyz'%(Lm,nx)
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
	fl.write('%1.3f %1.3f %1.3f \n'%(0,superprimitive1_rot[0]+superprimitive2_rot[0],superprimitive2_rot[0]))
	fl.write('%1.3f %1.3f %1.3f \n'%(0,superprimitive2_rot[1],0))
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
	generate_displacement_field(12,13, 0.77)
	#generate_displacement_field(27,28, 0.692)
	#generate_displacement_field(31,32, 0.61)
	#generate_displacement_field(31,32, 0.31)
