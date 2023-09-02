# Scaled variable

carregamento= 50000
E = 78e6
poisson = 0.3
lambda_ = E*poisson / ((1+poisson)*(1-2*poisson))
G= E / 2*(1+poisson)


import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx.io import gmshio
from dolfinx import mesh, fem, log

#Importação da geometria e das condições de contorno.

domain= mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([2,1])],
                  [400,400], cell_type=mesh.CellType.triangle)

V = fem.VectorFunctionSpace(domain, ("CG", 1))

#condições de contorno
def clamped_boundary(x):
    return np.isclose(x[0], 0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
u_D = np.array([0,0], dtype=ScalarType)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
T = fem.Constant(domain, ScalarType((0, 0)))

#especificar a medida de integração, que deve ser a integral sobre a fronteira do nosso domínio
ds = ufl.Measure("ds", domain=domain)

#Aproximação das funções teste e funcao incognita
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

def epsilon(x): #tensor para pequenas deformações
    return ((1/2)*((ufl.nabla_grad(x)) + (ufl.nabla_grad(x).T) ))

def sigma(y): 
    d = len(y)
    I = ufl.variable(ufl.Identity(d))
    return 2.0 * G * epsilon(y) + lambda_ * ufl.tr(epsilon(y)) * I 

#funcao carregamento
f = fem.Constant(domain, ScalarType((0,carregamento )))

#Formulação variacional 
a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
L = ufl.dot(f, v) * ds

problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

log.set_log_level(log.LogLevel.INFO)
uh = problem.solve()


from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "malha001.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
   # xdmf.write_meshtags(facet_tags)
   # xdmf.write_meshtags(cell_tags)
    xdmf.write_function(uh)



if __name__ == "__main__":
    pass

