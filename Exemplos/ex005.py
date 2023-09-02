from dolfinx import fem
import dolfinx
import ufl
from ufl import ds, dx, grad, inner, dot
from petsc4py.PETSc import ScalarType

from dolfinx import fem, mesh
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
u = ufl.TrialFunction(V)
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
epsilon = ((1/2)*((ufl.grad(u)) + (ufl.grad(u).T) ))
d = len(u)
I = ufl.Identity(d)

    
sigma = (2.0 * G * epsilon + lambda_ * ufl.tr(epsilon) * I )

#funcao carregamento
f = fem.Constant(domain, ScalarType((0,carregamento )))

#Formulação variacional 
a = ufl.inner(sigma, ufl.grad(v)) * ufl.dx
L = ufl.dot(f, v) * ds

problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()







from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "malha001.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
   # xdmf.write_meshtags(facet_tags)
   # xdmf.write_meshtags(cell_tags)
    xdmf.write_function(uh)

