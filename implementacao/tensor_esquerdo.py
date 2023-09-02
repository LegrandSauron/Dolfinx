from dolfinx import fem
import ufl
from ufl import ds, dx, grad, inner, dot
from petsc4py.PETSc import ScalarType
from dolfinx.io import XDMFFile
from dolfinx import fem, mesh
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc

malha= mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([2,1])],
                  [20,6,6], cell_type=mesh.CellType.triangle)

V = fem.VectorFunctionSpace(malha, ("CG", 1))


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

""" Faces para aplicação das condições de contorno
Physical Curve("esgaste", 16) = {14, 15, 3, 1};
Physical Curve("carregamento", 17) = {8};
Physical Surface("dominio", 18) = {1, 2, 3, 4};
"""

fdim= malha.topology.dim - 1
u_D = np.array([0,0], dtype=ScalarType)
dofs_2 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(16))
bc_2=fem.dirichletbc(u_D,dofs=dofs_2,V=V)
bcs=[bc_2]

K= 1
G=1
lambda_ = K - (2/3)*G


d = len(u)
I = ufl.variable(ufl.Identity(d)) #Identidade
F = ufl.variable(I + ufl.grad(u)) #Tensor de deformação



C = ufl.variable(F *F.T) # Tensor de cauchy-green esquerdo

#H= ufl.ln(C)

metadata = {"quadrature_degree": 4}
ds = ufl.Measure('ds', domain=malha, subdomain_data=facet_tags, metadata=metadata)
dx = ufl.Measure("dx", domain=malha, metadata=metadata)

# Elasticity parameters
E = PETSc.ScalarType(1.0e4)
nu = PETSc.ScalarType(0.3)
mu = fem.Constant(malha, E/(2*(1 + nu)))
lmbda = fem.Constant(malha, E*nu/((1 + nu)*(1 - 2*nu)))
#T = fem.Constant(malha, PETSc.ScalarType((0, 0)))
T = fem.Constant(malha, ScalarType((0,6)))


P = 2.0 * mu * ufl.sym(ufl.grad(u)) + lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * I #tensor de tensão

F = - ufl.inner(ufl.grad(v), T)* ufl.dx - ufl.inner(v, T)* ufl.ds(17)  #Forma variacional

#problem = fem.petsc.NonlinearProblem(F, u, bcs)
problem = fem.petsc.LinearProblem()

from dolfinx import nls
solver = nls.petsc.NewtonSolver(malha.comm, problem)

# Set Newton solver options
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"








