from dolfinx import fem
import ufl
from ufl import ds, dx, grad, inner, dot
from petsc4py.PETSc import ScalarType
from dolfinx.io import XDMFFile
from dolfinx import fem, mesh, log
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc

domain= mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([2,1])],
                  [400,200], cell_type=mesh.CellType.triangle)

V = fem.VectorFunctionSpace(domain, ("CG", 1))


def clamped_boundary(x):
    return np.isclose(x[0], 0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
u_D = np.array([0,0], dtype=ScalarType)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
T = fem.Constant(domain, ScalarType((0, 0)))
ds = ufl.Measure("ds", domain=domain)


u = fem.Function(V)
v = ufl.TestFunction(V)

""" Faces para aplicação das condições de contorno
Physical Curve("esgaste", 16) = {14, 15, 3, 1};
Physical Curve("carregamento", 17) = {8};
Physical Surface("dominio", 18) = {1, 2, 3, 4};
"""

# Scaled variable

carregamento= 50000
E = 78e6
poisson = 0.3
lambda_ = E*poisson / ((1+poisson)*(1-2*poisson))
G= E / 2*(1+poisson)


#hencky's strain tensor
d = len(u)
I = (ufl.Identity(d)) 
F =(I + ufl.grad(u)) 
C= ufl.dot(F, F.T )
N, M = C.ufl_shape
Hencky_Strain= ufl.as_tensor([[(1/2)*(ufl.ln(C[i,j])) for i in range(N)] for j in range(M)])


def epsilon(x): #tensor para pequenas deformações
    return ((1/2)*((ufl.nabla_grad(x)) + (ufl.nabla_grad(x).T) ))

def sigma(y): 
    d = len(y)
    I = ufl.variable(ufl.Identity(d))
    return 2.0 * G * epsilon(y) + lambda_ * ufl.tr(epsilon(y)) * I 




T_tensao= 2.0 * G * Hencky_Strain + lambda_ * ufl.tr(Hencky_Strain) * I 
#funcao carregamento
f = fem.Constant(domain, ScalarType((0,carregamento )))

#formulação fraca
#a = ufl.inner(ufl.grad(v),T_tensao) * ufl.dx -  ufl.inner(f, v) * ds


#Formulação variacional 
#a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
#L = ufl.dot(f, v) * ds

a = ufl.inner(T_tensao,ufl.grad(v)) * ufl.dx -  ufl.inner(f, v) * ds


#SOLUCAO
problem = fem.petsc.NonlinearProblem(F=a ,u=u,bcs=[bc])



from dolfinx import nls
solver = nls.petsc.NewtonSolver(domain.comm, problem)


#Set Newton solver options
solver.atol = 1e-12
solver.rtol = 1e-12
solver.convergence_criterion = "incremental"
solver.report = True

#config
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()



log.set_log_level(log.LogLevel.INFO)
n, converged = solver.solve(u=u)
assert(converged)
print(f"Number of interations: {n:d}")



from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "exemplo_linear_elasticity.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
   # xdmf.write_meshtags(facet_tags)
   # xdmf.write_meshtags(cell_tags)
  # xdmf.write_function(solver)

