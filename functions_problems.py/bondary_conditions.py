"""Definindo condições iniciais para elementos mistos"""

import numpy as np
import ufl
from dolfinx.io import XDMFFile
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, nls

domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([2,1])],
                  [40,40], cell_type=mesh.CellType.triangle)

x = ufl.SpatialCoordinate(domain)

"""time parameters"""
t_init= 0.0
steps=100
Tempo_final= 1
dt= fem.Constant(domain,Tempo_final/steps)

kappa= fem.Constant(domain,237e-6)  # thermal conductivity


"""espaços de funcoes"""
# Define function space, scalar
U2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)  # Displacent
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)  # temperature

# DOFs
TH = ufl.MixedElement([U2, P1])
V = fem.FunctionSpace(domain, TH)  # Total space for all DOFs

# Define actual functions with the required DOFs
w = ufl.TrialFunction(V)
#w = fem.Function(V)
(u, Temperatura) = ufl.split(w)  # current DOFs
(u_test, T_test) = ufl.TestFunctions(V)  # Test function

"""Definindo as condições de contorno"""
boundaries = [
    (1, lambda x: np.isclose(x[0], 0)),
    (2, lambda x: np.isclose(x[0], 2 )),
    (3, lambda x: np.isclose(x[1], 0)),
    (4, lambda x: np.isclose(x[1],1))]
  
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

"""Funcoes de interpolacao"""
def fixed_displacement_expression(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def incremented_displacement_expression(x):
    return np.full(x.shape[1], 297.0 ) #1.0e-02

V0, submap1 = V.sub(0).collapse()

fixed_displacement1 = fem.Function(V0)
fixed_displacement1.interpolate(fixed_displacement_expression)
left_dof = fem.locate_dofs_topological((V.sub(0),V0), fdim, facet_tag.find(1))
bc1 = fem.dirichletbc(fixed_displacement1, left_dof, V.sub(0))

VS, submap2 = V.sub(1).collapse()
fixed_displacement2 = fem.Function(VS)
fixed_displacement2.interpolate(incremented_displacement_expression)
left_dof = fem.locate_dofs_topological((V.sub(1),VS), fdim, facet_tag.find(1))
bc2 = fem.dirichletbc(fixed_displacement2, left_dof, V.sub(1))

bc= [bc2]

"""Solucao inicial"""
u0 = fem.Function(V)
c0, T_init = ufl.split(u0)

#u0.sub(1).interpolate(incremented_displacement_expression)
u0.sub(1).interpolate(lambda x: np.full(x.shape[1], 0.00))
u0.x.scatter_forward()


#Salvando os dados 
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "resultados/boundary_conditions.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

            
#Defining the body force term
q = fem.Constant(domain, PETSc.ScalarType(0.0))

#Forma variacional
a     = ufl.inner(T_test,Temperatura)*ufl.dx + dt*ufl.inner(ufl.grad(T_test),ufl.grad(Temperatura))*kappa*ufl.dx
L     = dt*ufl.inner(T_test,q)*ufl.dx + ufl.inner(T_test,T_init)*ufl.dx

# Solve the linear system
problem = fem.petsc.LinearProblem(a, L, bcs = bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})

for i in range(steps):
    t_init += dt
    uh 	  = problem.solve()
    u0.sub(1).interpolate(uh.sub(1))
    #T_init.x.scatter_forward()
    xdmf.write_function(uh.sub(1),t_init)


