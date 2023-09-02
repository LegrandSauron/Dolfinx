from dolfinx import fem
import ufl
from petsc4py.PETSc import ScalarType
from dolfinx import fem,log, nls ,mesh, plot
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc

# Scaled variable
carregamento= -5000
E = 78e6
poisson = 0.3
lambda_ = E*poisson / ((1+poisson)*(1-2*poisson))
G= E / 2*(1+poisson)



domain= mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([2,1])],
                  [100,100], cell_type=mesh.CellType.triangle)


V = fem.VectorFunctionSpace(domain, ("CG", 1))

def left(x):
    return np.isclose(x[0], 0)


fdim = domain.topology.dim -1
left_facets = mesh.locate_entities_boundary(domain, fdim, left)

# Concatenate and sort the arrays based on facet indices. Left facets marked with 1, right facets with two
marked_facets = np.hstack([left_facets])
marked_values = np.hstack([np.full_like(left_facets, 1)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

u_bc = np.array((0,) * domain.geometry.dim, dtype=PETSc.ScalarType)

left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bcs = [fem.dirichletbc(u_bc, left_dofs, V)]

B = fem.Constant(domain, PETSc.ScalarType((0, 0, 0)))


#Aproximação das funções teste e funcao incognita
v = ufl.TestFunction(V)
u= fem.Function(V)

#hencky's strain tensor
#d = len(u)
#I = (ufl.Identity(d)) 
#F =(I + ufl.grad(u)) 

# Spatial dimension
d = len(u)
# Identity tensor
I = ufl.variable(ufl.Identity(d))
# Deformation gradient
F = ufl.variable(I + ufl.grad(u))
# Right Cauchy-Green tensor
C = ufl.variable(F * F.T)
# Invariants of deformation tensors
Ic = ufl.variable(ufl.tr(C))
Jhc_e  = ufl.variable(ufl.det(F))
J= Jhc_e

#Hencky's Strain
N, M = C.ufl_shape
T_hencky=ufl.as_tensor([[(1/2)*(ufl.ln(C[i,j])) for i in range(N)] for j in range(M)])


#Tensor de tensões
T_tension= (1/J)*(2.0 * G * T_hencky + lambda_ * ufl.tr(T_hencky) * I )


#Funcao carregamento = f(0,y)
f = fem.Constant(domain, ScalarType((0,carregamento )))

metadata = {"quadrature_degree": 4}
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

#Weak form for  Div T = 0,  Tn= f, u= 0 in 
F_bilinear = ufl.inner(ufl.grad(v),T_tension) * dx - ufl.inner(f, v) *ds

problem = fem.petsc.NonlinearProblem(F_bilinear, u, bcs)
solver = nls.petsc.NewtonSolver(domain.comm, problem)

# Set Newton solver options
solver.atol = 1e-8
solver.rtol = 1e-6
solver.convergence_criterion = "residual"
solver.krylov_solver


log.set_log_level(log.LogLevel.INFO)
num_its, converged = solver.solve(u)
assert(converged)
