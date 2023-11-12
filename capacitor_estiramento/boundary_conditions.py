
from mpi4py import MPI
from dolfinx import fem, io, nls, log, mesh, plot
import numpy as np
import pyvista
from ufl import VectorElement,FiniteElement,MixedElement,TestFunction,TrialFunction,split,grad,tr,Identity,inner,dot
from petsc4py.PETSc import ScalarType
import ufl
from dolfinx.io import gmshio

domain, cell_tags, facet_tags = gmshio.read_from_msh("capacitor_malha.msh", MPI.COMM_SELF,0, gdim=2)

"""Function Space"""

# Define function space, scalar
U2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)  # Displacent
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)  # concentrações + - , electric potential
D0 = ufl.FiniteElement("DG", domain.ufl_cell(), 0)
D1 = ufl.FiniteElement("DG", domain.ufl_cell(), 1)

# DOFs
TH = ufl.MixedElement([U2, P1, P1, P1])
ME = fem.FunctionSpace(domain, TH)  # Total space for all DOFs

"""Extraindo os sub espaços do elemento misto e os mapas contendo os graus de liberdade """
num_subs = ME.num_sub_spaces
spaces = []
maps = []
for i in range(num_subs):
    space_i, map_i = ME.sub(i).collapse()
    spaces.append(space_i)
    maps.append(map_i)

(u_test, omgPos_test, phi_test, omgNeg_test)  = ufl.TestFunctions(ME)    # Test function 

W = TrialFunction(ME)


"""Bondary condition """
fdim = domain.topology.dim - 1

def fixed_displacement_expression(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

fixed_displacement = fem.Function(spaces[0])
fixed_displacement.interpolate(fixed_displacement_expression)

Engast_0 = fem.locate_dofs_topological((ME.sub(0), spaces[0]), fdim, facet_tags.find(14))
Enga0 = fem.dirichletbc(fixed_displacement, Engast_0, ME.sub(0))

#Aterramento e Voltagem eletrica
"""
1 14 "eletrodo_superior_l"
1 15 "gel_superior_l"
1 16 "gel_inferior_l"
1 17 "eletrodo_inferior_l"

1 18 "eletrodo_inferior_r"
1 19 "gel_inferior_r"
1 20 "gel_superior_r"
1 21 "eletrodo_superior_r"

1 29 "eletrodo_sup_cima"
1 30 "eletrodo_inf_baixo"

2 26 "eletrodo_superior"
2 27 "eletrodo_inferior"
2 28 "gel"
"""

"""Engaste"""
Engast_0 = fem.locate_dofs_topological((ME.sub(0), spaces[0]), fdim, facet_tags.find(14))
Engast0 = fem.dirichletbc(fixed_displacement, Engast_0, ME.sub(0))

Engast_1 = fem.locate_dofs_topological((ME.sub(0), spaces[0]), fdim, facet_tags.find(15))
Engast1 = fem.dirichletbc(fixed_displacement, Engast_1, ME.sub(0))

Engast_2 = fem.locate_dofs_topological((ME.sub(0), spaces[0]), fdim, facet_tags.find(16))
Engast2 = fem.dirichletbc(fixed_displacement, Engast_2, ME.sub(0))

Engast_3 = fem.locate_dofs_topological((ME.sub(0), spaces[0]), fdim, facet_tags.find(17))
Engast3 = fem.dirichletbc(fixed_displacement, Engast_3, ME.sub(0))

"""Estiramento : Deve-se alterar para obter os valores ideais de estiramento"""
scaleX = 1.0e4
T2_tot = 30.0e6 #0.0001*1.e6 #t1+t2+t3+t4

class disp_Exp:
    def __init__(self):
        self.t= 0.0
        self.Tramp =T2_tot
        self.scalex = scaleX
    
    def eval(self,x):
        return np.stack((np.zeros(x.shape[1]), np.full(x.shape[1], 5.5*self.scalex*self.t/self.Tramp)))

dispV= disp_Exp()
dispV.t= 0.
dispV.Tramp= T2_tot
dispV.scalex= scaleX
disp= fem.Function(spaces[0])
disp.interpolate(dispV.eval)

stretch00= fem.locate_dofs_topological((ME.sub(0).sub(0),spaces[0]),fdim,facet_tags.find(18)) 
bc_stretch0= fem.dirichletbc(disp, stretch00, ME.sub(0))

stretch01= fem.locate_dofs_topological((ME.sub(0).sub(0),spaces[0]),fdim,facet_tags.find(19)) 
bc_stretch1= fem.dirichletbc(disp, stretch01, ME.sub(0))

stretch02= fem.locate_dofs_topological((ME.sub(0).sub(0),spaces[0]),fdim,facet_tags.find(20)) 
bc_stretch2= fem.dirichletbc(disp, stretch02, ME.sub(0))

stretch03= fem.locate_dofs_topological((ME.sub(0).sub(0),spaces[0]),fdim,facet_tags.find(21)) 
bc_stretch3= fem.dirichletbc(disp, stretch03, ME.sub(0))


"""Aterramento"""
def ground_0(x):
    return np.stack((np.zeros(x.shape[1])))

ground = fem.Function(spaces[2])
ground.interpolate(ground_0)

ground0 = fem.locate_dofs_topological((ME.sub(2), spaces[2]), fdim, facet_tags.find(30))
bc_ground = fem.dirichletbc(ground, ground0, ME.sub(2))


"""Aplicação da voltagem inicial de 0 à 1, em um intervalo de 30s, e permanecer constante em 1 indefinidamente """
RT = 8.3145e-3*(273.0+20.0)      # Gas constant*Temp [ML2T-2#-1]
Farad = 96485.e-6  
phi_norm = RT/Farad # "Thermal" Volt

class phiRamp_function():
    def __init__(self):
        self.phi_norm= phi_norm
        self.pi=np.pi
        self.Tramp= 30.0e6
        self.t= 0.0
        self.temp= 1.0e3
        
    def phi_eval(self,x):
        return np.stack((np.full(x.shape[1], min(self.temp/self.phi_norm*(self.t/self.Tramp), self.temp/self.phi_norm))))

phiRamp_func= phiRamp_function()
phiRamp_func.t= 0 
phiRamp= fem.Function(spaces[2])
phiRamp.interpolate(phiRamp_func.phi_eval)

phiRamp_0= fem.locate_dofs_topological((ME.sub(2),spaces[2]),fdim,facet_tags.find(29))
bc_phiRamp = fem.dirichletbc(phiRamp,phiRamp_0,ME.sub(2))

bc=[Engast0,Engast1,Engast2,Engast3, bc_stretch0,bc_stretch1,bc_stretch2,bc_stretch3, bc_ground,bc_phiRamp  ]

