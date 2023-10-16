import gmsh
import numpy as np
import pyvista
from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, locate_entities
from dolfinx.plot import create_vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
                 dx, grad, inner)
import ufl 
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import fem, mesh
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([2,1])],
                  [40,40], cell_type=mesh.CellType.triangle)

Q = FunctionSpace(domain, ("DG", 0))

# Scaled variable
Temp_init = 300.0
Temp_last =25.0

#time parameters
t_init= 0.0
steps=100
Tempo_final= 1
dt= fem.Constant(domain,Tempo_final/steps)

#definindo o espaço de funções 
V=fem.FunctionSpace(domain, ("CG", 1))



#Solucao inicial:
u_init = fem.Function(V)
def initial_condition(x, a=0.0):
    return np.exp(-a*(x[0])+ a*x[1])
u_init.interpolate(initial_condition)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)




#material diferente
def Omega_0(x):
    return x[1] <= 0.5

def Omega_1(x):
    return x[1] >= 0.5


kappa = Function(Q)
cells_0 = locate_entities(domain, domain.topology.dim, Omega_0)
cells_1 = locate_entities(domain, domain.topology.dim, Omega_1)

kappa.x.array[cells_0] = np.full_like(cells_0, 0.5, dtype=ScalarType)
kappa.x.array[cells_1] = np.full_like(cells_1, 0.1, dtype=ScalarType)



boundaries = [
    (1, lambda x: np.isclose(x[0], 0)),
    (2, lambda x: np.isclose(x[0], 2 )),
    (3, lambda x: np.isclose(x[1], 0)),
    (4, lambda x: np.isclose(x[1],1))
    
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

left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bc = [fem.dirichletbc(ScalarType(Temp_init), left_dofs, V)]






x = SpatialCoordinate(domain)

#Salvando os dados 
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm,"resultados/tcmsub.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
        
#Defining the body force term
q = fem.Constant(domain, ScalarType(0.0))

#Forma variacional
a     = ufl.inner(v,u)*ufl.dx + dt*ufl.inner(ufl.grad(v),ufl.grad(u))*kappa*ufl.dx
L     = dt*ufl.inner(v,q)*ufl.dx + ufl.inner(v,u_init)*ufl.dx

# Solve the linear system
problem = fem.petsc.LinearProblem(a, L, bcs = bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})

for i in range(steps):
    t_init += dt
    uh 	  = problem.solve()
    u_init.interpolate(uh)
    xdmf.write_function(uh,t_init)



