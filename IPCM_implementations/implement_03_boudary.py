
from mpi4py import MPI
from dolfinx import fem, io, nls, log, mesh, plot
import numpy as np
import pyvista
from ufl import VectorElement,FiniteElement,MixedElement,TestFunction,TrialFunction,split,grad,tr,Identity,inner,dot
from petsc4py.PETSc import ScalarType
import ufl
from dolfinx.io import gmshio

domain, cell_tags, facet_tags = gmshio.read_from_msh("capacitor_malha.msh", MPI.COMM_SELF,0, gdim=2)

"""Parameters"""
T0 = fem.Constant(domain,293.)
DThole = fem.Constant(domain,10.)
E = 70e3
nu = 0.3
lmbda = fem.Constant(domain,E*nu/((1+nu)*(1-2*nu)))
mu = fem.Constant(domain,E/2/(1+nu))
rho = 2700.0    # density
alpha = 20.31e-6  # thermal expansion coefficient
kappa = alpha*(2*mu + 3*lmbda)
cV = fem.Constant(domain,910e-6)*rho  # specific heat per unit volume at constant strain
k = fem.Constant(domain,237e-6)  # thermal conductivity

"""Function Space"""

# Define function space, scalar
U2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)  # Displacent
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)  # concentrações + - , electric potential
D0 = ufl.FiniteElement("DG", domain.ufl_cell(), 0)
D1 = ufl.FiniteElement("DG", domain.ufl_cell(), 1)

# DOFs
TH = ufl.MixedElement([U2, P1, P1, P1])
ME = fem.FunctionSpace(domain, TH)  # Total space for all DOFs

#Extraindo os sub espaços do elemento misto e os mapas contendo os graus de liberdade 
num_subs = ME.num_sub_spaces
spaces = []
maps = []
for i in range(num_subs):
    space_i, map_i = ME.sub(i).collapse()
    spaces.append(space_i)
    maps.append(map_i)

(u_, Theta_) = ufl.TestFunctions(ME)

W = TrialFunction(ME)

"""Bondary condition """



u_D0 = fem.Function(spaces[0])
u_D0.x.array[:] = 0.0 #Teste usando np.arange para implementar a matriz de valores da condição de contorno

u_D1 = fem.Function(spaces[1])
u_D1.x.array[:] = DThole

fdim = domain.topology.dim - 1


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

#Engaste
Engast_0 = fem.locate_dofs_topological((ME.sub(0), spaces[0]), fdim, facet_tags.find(14))
Engast0 = fem.dirichletbc(u_D0, Engast_0, ME.sub(0))

Engast_1 = fem.locate_dofs_topological((ME.sub(0), spaces[0]), fdim, facet_tags.find(15))
Engast1 = fem.dirichletbc(u_D0, Engast_1, ME.sub(0))

Engast_2 = fem.locate_dofs_topological((ME.sub(0), spaces[0]), fdim, facet_tags.find(16))
Engast2 = fem.dirichletbc(u_D0, Engast_2, ME.sub(0))

Engast_3 = fem.locate_dofs_topological((ME.sub(0), spaces[0]), fdim, facet_tags.find(17))
Engast3 = fem.dirichletbc(u_D0, Engast_3, ME.sub(0))

#Estiramento
disp = Expression(("5.5*scaleX*t/Tramp"),
                  scaleX = scaleX, Tramp = 30.0e6, t = 0.0, degree=1)
    
    
bc_stretch0= fem.dirichletbc((ME.sub(0).sub(0),spaces[0]),disp,facet_tags.find(18)) 
bc_stretch1= fem.dirichletbc((ME.sub(0).sub(0),spaces[0]),disp,facet_tags.find(19)) 
bc_stretch2= fem.dirichletbc((ME.sub(0).sub(0),spaces[0]),disp,facet_tags.find(20)) 
bc_stretch3= fem.dirichletbc((ME.sub(0).sub(0),spaces[0]),disp,facet_tags.find(21)) 


#Aterramento
bc_ground = fem.dirichletbc((ME.sub(2),spaces[2]), 0.0 ,facet_tags.find(30)) # Ground bottom of device 

#Voltagem
phiRamp = Expression(("min(1.0e3/phi_norm*(t/Tramp), 1.0e3/phi_norm)"),
                 phi_norm = phi_norm, pi=np.pi, Tramp = 30.0e6, t = 0.0, degree=2)

bcs_f = fem.dirichletbc((ME.sub(2),spaces[2]),phiRamp,facet_tags.find(29)) #Aplicação da voltagem
