from dolfinx import fem
import dolfinx
import ufl
from ufl import ds, dx, grad, inner, dot
from petsc4py.PETSc import ScalarType
from dolfinx.io import XDMFFile
from dolfinx import fem, mesh, log, nls
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc 
from petsc4py.PETSc import*
import petsc4py

# Scaled variable
carregamento= 5000
E = 78e6
poisson = 0.3
lambda_ = E*poisson / ((1+poisson)*(1-2*poisson))
G= E / 2*(1+poisson)


domain, cell_tags, facet_tags = gmshio.read_from_msh("malha_com_eletrodo_04.msh", MPI.COMM_WORLD,0, gdim=3)

#Realizando a aproximação do dominio por um espaço de funcao V  
V = fem.VectorFunctionSpace(domain, ("CG", 1))

Q = fem.FunctionSpace(domain, ("DG", 0))

Emod = fem.Function(Q)
eletrodo_sup = cell_tags.find(26)
Emod.x.array[eletrodo_sup] = np.full_like(eletrodo_sup, 1, dtype=PETSc.ScalarType)
eletrodo_inf = cell_tags.find(27)
Emod.x.array[eletrodo_inf]  = np.full_like(eletrodo_inf, 1, dtype=PETSc.ScalarType)
gel_p=cell_tags.find(28)
Emod.x.array[gel_p]  = np.full_like(gel_p, 0.1, dtype=PETSc.ScalarType)


dofs_1 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(14))
dofs_2 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(15))
dofs_3 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(16))
dofs_4 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(17))


u_bc = np.array((0,) * domain.geometry.dim, dtype=PETSc.ScalarType)
bcs_1 = fem.dirichletbc( u_bc, dofs_1, V)
bcs_2 = fem.dirichletbc( u_bc, dofs_2, V)
bcs_3 = fem.dirichletbc( u_bc, dofs_3, V)
bcs_4 = fem.dirichletbc( u_bc, dofs_4, V)
bcs= [bcs_1,bcs_2,bcs_3,bcs_4]


#Aproximação das funções teste e funcao incognita
v = ufl.TestFunction(V)
u= ufl.TrialFunction(V)


# Spatial dimension
d = len(u)

# Identity tensor
I = ufl.variable(ufl.Identity(d))

# Deformation gradient
F = ufl.variable(I + ufl.grad(u))

# Right Cauchy-Green tensor
C = ufl.outer(F , F.T)


def epsilon(x): #tensor para pequenas deformações
    return ((1/2)*((ufl.nabla_grad(x)) + (ufl.nabla_grad(x).T) ))


def sigma(y): 
    d = len(y)
    I = ufl.variable(ufl.Identity(d))
    return 2.0 * G * epsilon(y) + lambda_ * ufl.tr(epsilon(y)) * I 

#funcao carregamento
f = fem.Constant(domain, ScalarType((0,carregamento)))

#Formulação variacional 
a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
L = ufl.dot(f,v)* ufl.dx

problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
