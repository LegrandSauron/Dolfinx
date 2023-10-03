


import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.fem import FunctionSpace, Function, Constant
from petsc4py.PETSc import ScalarType
import numpy as np

#import matplotlib.pyplot as plt

#Parâmetros
L_comprimento     = 2.0
A     = 1
uD    = 0.0
C     = 1
E     = 10E+4
t_bar = -C*L_comprimento**2/A

dominio	= mesh.create_interval(MPI.COMM_WORLD, nx=20, points=(0, L_comprimento))
V		= FunctionSpace(dominio,("CG",1))

def contorno_uD(x):
	return(np.isclose(x[0], 0, atol=1e-15))
 
# Create facet to cell connectivity required to determine boundary facets
tdim = dominio.topology.dim
fdim = tdim - 1
dominio.topology.create_connectivity(fdim, tdim)
esquerda_faceta_contorno	= mesh.locate_entities_boundary(dominio,fdim,contorno_uD)
esquerda_contorno_gdl 	  	= fem.locate_dofs_topological(V, fdim, esquerda_faceta_contorno)
uD_bc_esquerda				= fem.dirichletbc(ScalarType(uD), esquerda_contorno_gdl, V)

#Definindo funções tentativa e teste
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

#Defining the body force term
b = fem.Constant(dominio, ScalarType(1.0))
t_bar = fem.Constant(dominio, ScalarType(C*L_comprimento/A))

#Forma variacional
a = ufl.dot(ufl.grad(u),ufl.grad(v))*ufl.dx
L = ufl.dot(t_bar,v)*ufl.dx

# Solve the linear system
problem = fem.petsc.LinearProblem(a, L, bcs = [uD_bc_esquerda], petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
uh 		= problem.solve()

#Defining plot paraviews
xdmf = io.XDMFFile(dominio.comm, "bar_elastica.xdmf", "w")
xdmf.write_mesh(dominio)
xdmf.write_function(uh)
