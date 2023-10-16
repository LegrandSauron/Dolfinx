import ufl
from petsc4py.PETSc import ScalarType

from dolfinx import fem, mesh, plot
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc 
from petsc4py.PETSc import*

# Scaled variable
L_comprimento = 0.5
A      = 0.1
T_init = 30.0
k      = .005
T_last = 0.0

domain= mesh.create_interval(MPI.COMM_WORLD,20,np.array([0,L_comprimento]) )
x = ufl.SpatialCoordinate(domain)



# Define function space, scalar
U2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)  # Displacent
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)  # temperature

# DOFs
TH = ufl.MixedElement([U2, P1])
V = fem.FunctionSpace(domain, TH)  # Total space for all DOFs

(u_test, T_test) = ufl.TestFunctions(V)  # Test function

# Define actual functions with the required DOFs
w = ufl.TrialFunction(V)
(u, Temperatura) = ufl.split(w)  # current DOFs





#definindo o espaço de funções 



boundaries = [
    (1, lambda x: np.isclose(x[0], 0)),
    (2, lambda x: np.isclose(x[0], L_comprimento)),
    
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

"""Definindo as condições de contorno"""
def fixed_displacement_expression(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def incremented_displacement_expression(x):
    return np.full(x.shape[1], T_init ) #1.0e-02


V0, submap1 = V.sub(1).collapse()

Temperatura_left= fem.Function(V0)
Temperatura_left.interpolate(incremented_displacement_expression)
Temperatura_left.x.scatter_forward()

left_dof = fem.locate_dofs_topological((V.sub(1),V0), fdim, facet_tag.find(1))
bc1 = fem.dirichletbc(Temperatura_left, left_dof, V.sub(1))

bc= [bc1]
#Defining the body force term
s     = fem.Constant(domain, ScalarType(0.0))
q = fem.Constant(domain, ScalarType(0.0))

#Forma variacional
a     = ufl.inner(ufl.grad(Temperatura),ufl.grad(T_test))*k*A*ufl.dx
L     = ufl.inner(q,T_test)*k*A*ds(2) + ufl.inner(s,T_test)*ufl.dx

# Solve the linear system
problem = fem.petsc.LinearProblem(a, L, bcs = bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
uh 	  = problem.solve()

T_f= uh.sub(0).collapse()
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "resultados/acoplamento.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(T_f)

