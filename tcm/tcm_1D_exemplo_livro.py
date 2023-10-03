import ufl
from petsc4py.PETSc import ScalarType

from dolfinx import fem, mesh, plot
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc 
from petsc4py.PETSc import*

# Scaled variable
L_comprimento = 4.0
A      = 0.1
T_init = 0.0
k      = 2.0



dominio= mesh.create_interval(MPI.COMM_WORLD,20,np.array([0,L_comprimento]) )
x = ufl.SpatialCoordinate(dominio)

#definindo o espaço de funções 
V=fem.FunctionSpace(dominio, ("CG", 1))

#u = fem.Function(V)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)


boundaries = [
    (1, lambda x: np.isclose(x[0], 0)),
    (2, lambda x: np.isclose(x[0], L_comprimento)),
    
]
  
facet_indices, facet_markers = [], [] #matriz para criação dos indices de cada face e a face em si
fdim = dominio.topology.dim - 1

for (marker, locator) in boundaries:  #função percorre a lista boudaries, onde (marker = indices (1,2,3), locator= faces
    facets = mesh.locate_entities(dominio, fdim, locator) #coloca as faces (locator) dentro da variavel faces
    facet_indices.append(facets) #
    facet_markers.append(np.full_like(facets, marker))

facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)

facet_tag = mesh.meshtags(dominio, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

ds=ufl.Measure("ds",domain=dominio,subdomain_data=facet_tag) #definindo os subdominios de integração.

"""Definindo as condições de contorno"""
tdim = dominio.topology.dim
dominio.topology.create_connectivity(fdim, tdim)
esquerda_contorno_gdl 	  	= fem.locate_dofs_topological(V, fdim, facet_tag.find(1))
uD_bc_esquerda				= fem.dirichletbc(ScalarType(T_init), esquerda_contorno_gdl, V)


#Defining the body force term
s     = fem.Constant(dominio, ScalarType(5.0))
q = fem.Constant(dominio, ScalarType(-5.0))

#Forma variacional
a     = ufl.inner(ufl.grad(u),ufl.grad(v))*k*A*ufl.dx
L     = ufl.inner(q,v)*k*A*ds(2) + ufl.inner(s,v)*ufl.dx

# Solve the linear system
problem = fem.petsc.LinearProblem(a, L, bcs = [uD_bc_esquerda], petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
uh 	  = problem.solve()


from dolfinx.io import XDMFFile
with XDMFFile(dominio.comm, "resultados/acoplamento.xdmf", "w") as xdmf:
    xdmf.write_mesh(dominio)
   # xdmf.write_meshtags(facet_tags)
   # xdmf.write_meshtags(cell_tags)
    xdmf.write_function(uh)

