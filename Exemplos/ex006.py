from dolfinx import fem
import dolfinx
import ufl
from ufl import ds, dx, grad, inner, dot
from petsc4py.PETSc import ScalarType

from dolfinx import fem, mesh, nls, log
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc 
from petsc4py.PETSc import*
import petsc4py


# Scaled variable
carregamento= 50000
E = 78e6
poisson = 0.3
lambda_ = E*poisson / ((1+poisson)*(1-2*poisson))
G= E / 2*(1+poisson)


#criando a geometria e o numero de elementos
domain= mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0,0.0]), np.array([1.0, 1.0])],
                  [20,20], cell_type=mesh.CellType.triangle)

x = ufl.SpatialCoordinate(domain)

#definindo o espaço de funções 
V=fem.VectorFunctionSpace(domain, ("CG", 1))

#u = fem.Function(V)
du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)


boundaries = [
    (1, lambda x: np.isclose(x[0], 0)),
    (2, lambda x: np.isclose(x[0], 1)),
    (3, lambda x: np.isclose(x[1], 0)),
    (4, lambda x: np.isclose(x[1], 1))
]
  
facet_indices, facet_markers = [], [] #matriz para criação dos indices de cada face e a face em si
fdim = domain.topology.dim - 1

for (marker, locator) in boundaries:  #função percorre a lista boudaries, onde (marker = indices (1,2,3), locator= faces
    facets = mesh.locate_entities(domain, fdim, locator) #coloca as faces (locator) dentro da variavel faces
    facet_indices.append(facets) #
    facet_markers.append(np.full_like(facets, marker))

facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)

facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

ds=ufl.Measure("ds",domain=domain,subdomain_data=facet_tag) #definindo os subdominios de integração.

#COndções de contorno
u_D = np.array([0,0], dtype=ScalarType) 
dofs_2 = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bc=fem.dirichletbc(u_D,dofs=dofs_2,V=V)




#tensores
epsilon = ((1/2)*((ufl.grad(du)) + (ufl.grad(du).T) ))

d = len(du)
I = ufl.Identity(d)

T_d= I + ufl.grad(du)
C= T_d * T_d.T
N, M = C.ufl_shape
Hencky_Strain= ufl.as_tensor([[(1/2)*(ufl.ln(C[i,j])) for i in range(N)] for j in range(M)])


J= ufl.det(T_d)

sigma = (1/J)*(2.0 * G * Hencky_Strain + lambda_ * ufl.tr(Hencky_Strain) * I )


#funcao carregamento
f = fem.Constant(domain, ScalarType((0,carregamento )))

#Formulação variacional bilinear

a =  ufl.inner(ufl.grad(v),sigma) * ufl.dx + ufl.dot(f, v) * ds

#L = ufl.dot(f, v) * ds

#problem = fem.petsc.NonlinearProblem(a, u, bcs=[bc])


"""Solver para u= ufl.trialFunctionSpace(V)"""

#bilinear_form = fem.form(a)

#b = fem.petsc.create_vector(bilinear_form)
#A = fem.petsc.assemble_matrix(bilinear_form, bcs=[bc])
#A.assemble()



"""Solver para u = fem.function(V), realiza algumas interações, mas não congerve 

solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-15
solver.report = True

log.set_log_level(log.LogLevel.INFO)
n, converged = solver.solve(u)
assert(converged)
print(f"Number of interations: {n:d}")

"""


lalal, L = ufl.lhs(a), ufl.rhs(a)