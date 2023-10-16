import numpy as np
import ufl
from dolfinx.io import XDMFFile
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, nls

domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([2,1])],
                  [10,10], cell_type=mesh.CellType.triangle)
x = ufl.SpatialCoordinate(domain)

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


"""Espaços de Funcoes"""
U2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)  # Displacent
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)  # temperature

TH = ufl.MixedElement([U2, P1])
V = fem.FunctionSpace(domain, TH)

(u_test, T_test) = ufl.TestFunctions(V)

w = ufl.TrialFunction(V)
(u, Temperatura) = ufl.split(w)


def fixed_displacement_expression(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def incremented_displacement_expression(x):
    return np.full(x.shape[1], 2.0 ) #1.0e-02


V0, submap1 = V.sub(1).collapse()

fixed_displacement = fem.Function(V0)
fixed_displacement.interpolate(incremented_displacement_expression)
left_dof = fem.locate_dofs_topological((V.sub(0),V0), fdim, facet_tag.find(1))
bc1 = fem.dirichletbc(fixed_displacement, left_dof, V.sub(0))



#Salvando os dados 
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "resultados/acopla.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
  
    xdmf.write_function(fixed_displacement)