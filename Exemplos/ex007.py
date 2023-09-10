# Scaled variable

carregamento= -50000
E = 210e6
poisson = 0.3
lambda_ = E*poisson / ((1+poisson)*(1-2*poisson))
G= E / 2*(1+poisson)


import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx.io import gmshio
from dolfinx import mesh, fem, plot, io

#Importação da geometria e das condições de contorno.
domain, cell_tags, facet_tags = gmshio.read_from_msh("malha_estruturada.msh", MPI.COMM_SELF,0, gdim=2)

"""Facets_tags numbers options:
1 9 "face superior"
1 10 "face_esquerda"
1 11 "face_inferior"
1 12 "face_direita"
2 13 "dominio"
"""
# Define function space
V = fem.VectorFunctionSpace(domain, ("CG", 1))

# Dirichlet boundary
u_D = np.array([0,0], dtype=ScalarType) 
dofs_2 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(10))
bc=fem.dirichletbc(u_D,dofs=dofs_2,V=V)


#especificar a medida de integração, que deve ser a integral sobre a fronteira do nosso domínio
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags)

# Define test functions in weak form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

def epsilon(x): #tensor deform small
    return ((1/2)*((ufl.nabla_grad(x)) + (ufl.nabla_grad(x).T) ))

def sigma(y): 
    d = len(y)
    I = ufl.variable(ufl.Identity(d))
    return 2.0 * G * epsilon(y) + lambda_ * ufl.tr(epsilon(y)) * I 

#funcao carregamento
f = fem.Constant(domain, ScalarType((0,carregamento )))

#Formulação variacional 
a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
L = ufl.dot(f, v) *ds(9)

problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "gmres", "pc_type": "lu","ksp_max_it": 1000}, )

uh = problem.solve()



from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "resultados/malha001.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tags)
    xdmf.write_meshtags(cell_tags)
    xdmf.write_function(uh)


