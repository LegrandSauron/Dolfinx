from dolfinx import fem
import ufl
from ufl import ds, dx, grad, inner, dot
from petsc4py.PETSc import ScalarType
from dolfinx.io import XDMFFile
from dolfinx import fem, mesh, log, nls
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc 

domain= mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([2,1])],
                  [100,50], cell_type=mesh.CellType.triangle)

V = fem.VectorFunctionSpace(domain, ("CG", 1))

#Definindo as condições de contorno
def clamped_boundary(x):
    return np.isclose(x[0], 0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
u_D = np.array([0,0], dtype=ScalarType)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
#T = fem.Constant(domain, ScalarType((0, 0)))

#Espaço de integração
ds = ufl.Measure("ds", domain=domain)


#Aproximação das funções teste e funcao incognita
v = ufl.TestFunction(V)
uh = fem.Function(V)

# Scaled variable
carregamento= 50000
E = 78e6
poisson = 0.3
lambda_ = E*poisson / ((1+poisson)*(1-2*poisson))
G= E / 2*(1+poisson)


#tensor de deslocamentos
d = len(uh)
I = (ufl.Identity(d)) 
F_u =(I + ufl.grad(uh)) 

C= ufl.dot(F_u, F_u.T )
N, M = C.ufl_shape


#hencky's strain tensor
Hencky_Strain= ufl.as_tensor([[(1/2)*(ufl.ln(C[i,j])) for i in range(N)] for j in range(M)])


T_tensao= (1/ufl.det(F_u))*(2.0 * G * Hencky_Strain + lambda_ * ufl.tr(Hencky_Strain) * I )

#funcao carregamento
f = fem.Constant(domain, ScalarType((0,carregamento )))

#formulação fraca
F = ufl.inner(ufl.grad(v),T_tensao) * ufl.dx -  ufl.inner(f, v) * ds


"""Solução do sistema não linear"""

residual = fem.form(F)
J = ufl.derivative(F, uh)
jacobian = fem.form(J)

OptDB = PETSc.Options()

