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

#malha, cell_tags, facet_tags = gmshio.read_from_msh("malha_ece.msh", MPI.COMM_SELF,0, gdim=2)


#Criando ou importando a geometria e realizando a discretização.
domain= mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([2,1])],
                  [400,400], cell_type=mesh.CellType.triangle)


V= fem.VectorFunctionSpace(domain, ("CG", 1))
u = fem.Function(V)
v = ufl.TestFunction(V)



#Definindo as condições de contorno
def clamped_boundary(x):
    return np.isclose(x[0], 0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
u_D = np.array([0,0], dtype=ScalarType)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
T = fem.Constant(domain, ScalarType((0, 0)))

#Espaço de integração no contorno da geometria 
ds = ufl.Measure("ds", domain=domain)



""" Faces para aplicação das condições de contorno
Physical Curve("esgaste", 16) = {14, 15, 3, 1};
Physical Curve("carregamento", 17) = {8};
Physical Surface("dominio", 18) = {1, 2, 3, 4};

fdim= malha.topology.dim - 1
u_D = np.array([0,0], dtype=ScalarType)
dofs_2 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(16))
bc_2=fem.dirichletbc(u_D,dofs=dofs_2,V=V)
bcs=[bc_2]
"""
K= 1
G=1
lambda_ = K - (2/3)*G


d = len(u)
I = ufl.variable(ufl.Identity(d)) #Identidade
F = ufl.variable(I + ufl.grad(u)) #Tensor de deformação



C = ufl.variable(F *F.T) # Tensor de cauchy-green esquerdo

#H= ufl.ln(C)

metadata = {"quadrature_degree": 4}
ds = ufl.Measure('ds', domain=domain)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# Elasticity parameters
E = PETSc.ScalarType(1.0e4)
nu = PETSc.ScalarType(0.3)
mu = fem.Constant(domain, E/(2*(1 + nu)))
lmbda = fem.Constant(domain, E*nu/((1 + nu)*(1 - 2*nu)))
#T = fem.Constant(malha, PETSc.ScalarType((0, 0)))
T = fem.Constant(domain, ScalarType((0,6)))


P = 2.0 * mu * ufl.sym(ufl.grad(u)) + lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * I #tensor de tensão

#F = ufl.inner(ufl.grad(v), P)* ufl.dx - ufl.inner(v, T)* ufl.ds  #Forma variacional





problem = fem.petsc.NonlinearProblem(F, u, [bc])
#problem = fem.petsc.LinearProblem()

from dolfinx import nls
solver = nls.petsc.NewtonSolver(domain.comm, problem)

solver.solve(u=u)
# Set Newton solver options
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"
#solver.solve(u=u.vector)


from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "exemplo_linear_elasticity.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
   # xdmf.write_meshtags(facet_tags)
   # xdmf.write_meshtags(cell_tags)
    xdmf.write_function(u)






