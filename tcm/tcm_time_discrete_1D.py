import ufl
from petsc4py.PETSc import ScalarType

from dolfinx import fem, mesh, plot
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc 
L_comprimento = 0.1
dominio= mesh.create_interval(MPI.COMM_WORLD,300,np.array([0,L_comprimento]) )
x = ufl.SpatialCoordinate(dominio)

# Scaled variable

Temp_init = 30
k      = .00005
T_last =0.0

#time parameters
t_init= fem.Constant(dominio,0.0)
steps=200
Tempo_final= 1.
dt= fem.Constant(dominio,Tempo_final/steps)

#definindo o espaço de funções 
V=fem.FunctionSpace(dominio, ("CG", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)


#Solucao inicial:
u_init = fem.Function(V)
def initial_condition(x, a=0.0):
    return np.exp(-a*(x[0]))
u_init.interpolate(initial_condition)


#Definindo as condições iniciais
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
uD_bc_esquerda				= fem.dirichletbc(ScalarType(Temp_init), esquerda_contorno_gdl, V)

direita_ = fem.locate_dofs_topological(V,fdim,facet_tag.find(2))
ud_direita = fem.dirichletbc(ScalarType(T_last), direita_, V)

bc= [ud_direita,uD_bc_esquerda]


#Salvando os dados 
from dolfinx.io import XDMFFile
with XDMFFile(dominio.comm, "resultados/tcm_time_discrete_1D.xdmf", "w") as xdmf:
    xdmf.write_mesh(dominio)
    
    
#Defining the body force term
#s     = fem.Constant(dominio, ScalarType(5.0))
q = fem.Constant(dominio, ScalarType(0.0))

#Forma variacional
a     = ufl.inner(v,u)*ufl.dx + dt*ufl.inner(ufl.grad(v),ufl.grad(u))*k*ufl.dx
L     = dt*ufl.inner(v,q)*ufl.dx + ufl.inner(v,u_init)*ufl.dx



# Solve the linear system
problem = fem.petsc.LinearProblem(a, L, bcs = bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})

for i in range(steps):
    t_init += dt
    uh 	  = problem.solve()
    u_init.interpolate(uh)
    xdmf.write_function(uh,t_init)



