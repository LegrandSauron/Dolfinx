
import ufl
from dolfinx import*
import numpy as np
import math
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
from dolfinx.io import gmshio
from petsc4py import PETSc

domain, ct, facet_tags = gmshio.read_from_msh("malha_estruturada_sem_eletrodo_75_line.msh", MPI.COMM_SELF,0, gdim=2)

"""Obtenção das faces para aplicação das condições de contorno são dadas por : 

 9 "face superior"
 10 "face_esquerda"
 11 "face_inferior"
 12 "face_direita"
 13 "dominio"

    facet_tags.find(n), onde n [9 a 13]
"""
subdomain_0 = facet_tags.find(9)
subdomain_1 = facet_tags.find(10)
subdomain_2 = facet_tags.find(11)
subdomain_3 = facet_tags.find(12)
subdomain_4 = facet_tags.find(13)


#subdomain_0.mark(materials, 0)
N= 1
Q = fem.FunctionSpace(domain, ("DG", 0))
material_tags = np.unique(ct.values)
print(ct.values)
mu = fem.Function(Q)
J = fem.Function(Q)

# As we only set some values in J, initialize all as 0
J.x.array[:] = 0
for tag in material_tags:
    cells = ct.find(tag)
    # Set values for mu
    if tag == 0:
        mu_ = 4 * np.pi*1e-7 # Vacuum
    elif tag == 1:
        mu_ = 1e-5 # Iron (This should really be 6.3e-3)
    else:
        mu_ = 1.26e-6 # Copper
    mu.x.array[cells] = np.full_like(cells, mu_, dtype=ScalarType)
    if tag in range(2, 2+N):
        J.x.array[cells] = np.full_like(cells, 1, dtype=ScalarType)
    elif tag in range(2+N, 2*N + 2):
        J.x.array[cells] = np.full_like(cells, -1, dtype=ScalarType)
     



V = fem.FunctionSpace(domain, ("CG", 1))
tdim = domain.topology.dim

facets = mesh.locate_entities_boundary(domain, tdim-1, lambda x: np.full(x.shape[1], True))
dofs = fem.locate_dofs_topological(V, tdim-1, facets)
bc = fem.dirichletbc(ScalarType(0), dofs, V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = (1 / mu) * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = J * v * ufl.dx

A_z = fem.Function(V)
problem = fem.petsc.LinearProblem(a, L, u=A_z, bcs=[bc])
uh=problem.solve()


W = fem.VectorFunctionSpace(domain, ("DG", 0))
B = fem.Function(W)
B_expr = fem.Expression(ufl.as_vector((A_z.dx(1), -A_z.dx(0))), W.element.interpolation_points())
B.interpolate(B_expr)

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "malha001.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tags)
    xdmf.write_meshtags(ct)
    xdmf.write_function(B)
