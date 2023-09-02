from dolfinx import fem
import ufl
from ufl import ds, dx, grad, inner, dot
from petsc4py.PETSc import ScalarType
from dolfinx.io import XDMFFile
from dolfinx import fem, mesh
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np

#Importação da geometria e das condições de contorno.
malha, cell_tags, facet_tags = gmshio.read_from_msh("malha_001.msh", MPI.COMM_SELF,0, gdim=2)


V= fem.VectorFunctionSpace(malha, ("CG", 1))
u = fem.Function(V)
v = ufl.TestFunction(V)

""" Faces para aplicação das condições de contorno
Physical Curve("esgaste", 16) = {14, 15, 3, 1};
Physical Curve("carregamento", 17) = {8};
Physical Surface("dominio", 18) = {1, 2, 3, 4};
"""

fdim= malha.topology.dim - 1
u_D = np.array([0,0], dtype=ScalarType)
dofs_2 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(16))
bc_2=fem.dirichletbc(u_D,dofs=dofs_2,V=V)
bcs_=[bc_2]



"""Constantes"""
K= 1
G=1
lambda_ = K - (2/3)*G

"""Tensor Hencky's logarithmic strain."""
def epsilon_hencky(u):  
    return ((ufl.Identity(len(u)) + ufl.nabla_grad(u)) * (ufl.transpose(ufl.Identity(len(u)) + ufl.nabla_grad(u))))


""" Tensor de tensões, linear hyperelastic stress–strain relation  """ 
def sigma(u):
    epsilon_u = epsilon_hencky(u)
    return (1 / ufl.det(epsilon_u)) * (2 * G * epsilon_u + lambda_ * ufl.tr(epsilon_u) * ufl.nabla_div(u) * ufl.Identity(len(u)))


c= epsilon_hencky(u) #teste
al = sigma(u) # teste

T = fem.Constant(malha, ScalarType((0,6)))


a = ufl.inner(grad(v),sigma(u)) * ufl.dx
L = ufl.inner(v,T)* ds



problem = fem.petsc.LinearProblem(a, L, bcs=bcs_, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

"""
with XDMFFile(malha.comm, "malha_refi.xdmf", "w") as xdmf:
    xdmf.write_mesh(malha)
    xdmf.write_meshtags(facet_tags)
    xdmf.write_meshtags(cell_tags)
    xdmf.write_function(uh)
"""