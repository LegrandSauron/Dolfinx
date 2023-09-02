# Scaled variable
L = 1
W = 0.2
mu = 1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma

import numpy as np
import ufl

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx.io import gmshio
from dolfinx import mesh, fem, plot, io


#Importação da geometria e das condições de contorno.
domain, cell_tags, facet_tags = gmshio.read_from_msh("malha_001.msh", MPI.COMM_SELF,0, gdim=2)

V = fem.VectorFunctionSpace(domain, ("CG", 1))

#condições de contorno
def clamped_boundary(x):
    return np.isclose(x[0], 0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

u_D = np.array([0,0], dtype=ScalarType)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

#Como queremos a tração sobre o limite restante a ser, criamos umdolfinx.Constant
T = fem.Constant(domain, ScalarType((0, 0 )))


#especificar a medida de integração, que deve ser a integral sobre a fronteira do nosso domínio
ds = ufl.Measure("ds", domain=domain)


#Formulação variacional 
def epsilon(u): #tensor para pequenas deformações 
    return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u): #tensor de tensões 
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2*mu*epsilon(u)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, ScalarType((0, -rho*g)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()


from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "exemplo_linear_elasticity.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tags)
    xdmf.write_meshtags(cell_tags)
    xdmf.write_function(uh)