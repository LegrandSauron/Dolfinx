from ufl.classes import *
from dolfinx import fem
import ufl
from petsc4py.PETSc import ScalarType
from dolfinx import fem,log, nls ,mesh, plot
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc



# Scaled variable
carregamento= 50000
E = 78e6
poisson = 0.3
lambda_ = E*poisson / ((1+poisson)*(1-2*poisson))
G = E / (2 * (1 + poisson))


# Concatenate and sort the arrays based on facet indices. Left facets marked with 1, right facets with two
domain= mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([2,1])],
                  [100,100], cell_type=mesh.CellType.triangle)

V = fem.VectorFunctionSpace(domain, ("CG", 1))

#condições de contorno
def clamped_boundary(x):
    return np.isclose(x[0], 0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
u_D = np.array([0,0], dtype=ScalarType)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

#Aproximação das funções teste e funcao incognita
u  = ufl.TrialFunction(V)
v  = ufl.TestFunction(V)
uh = fem.Function(V)


# Spatial dimension
d = len(uh)
# Identity tensor
I =(ufl.Identity(d))
# Deformation gradient
F = (I + ufl.nabla_grad(uh))
# Right Cauchy-Green tensor
C = (F * F.T)
# Invariants of deformation tensors
Ic = (ufl.tr(C))
Jhc_e  =(ufl.det(F))
J= Jhc_e

#Hencky's Strain
N, M = C.ufl_shape
T_hencky=ufl.as_tensor([[(1/2)*(ufl.ln(C[i,j])) for i in range(N)] for j in range(M)])


#Tensor de tensões
T_tension= (1/J)*(2.0 * G * T_hencky + lambda_ * ufl.tr(T_hencky) * I )

print("\n",T_hencky[0,0].ufl_operands)
print("\n",T_hencky.ufl_shape)
print("\n",T_hencky.ufl_free_indices)
print("\n",str(T_hencky[0,0]))


"""
#Funcao carregamento = f(0,y)
f = fem.Constant(domain,ScalarType((0,carregamento)))


ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag)
dx = ufl.Measure("dx", domain=domain)

#Weak form for  Div T = 0,  Tn= f, u= 0 in 
F_bilinear = ufl.inner(ufl.nabla_grad(v),T_tension) * dx    -ufl.inner(f, v)*ds


jacobian= ufl.derivative(F_bilinear,u_ ,u)
"""

