import numpy as np
import ufl
from dolfinx.io import XDMFFile
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, plot

domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([2,1])],
                  [40,40], cell_type=mesh.CellType.triangle)

#time parameters
t_init= 0.0
steps=100
Tempo_final= 1
dt= fem.Constant(domain,Tempo_final/steps)



V = fem.VectorFunctionSpace(domain, ("CG", 1))


"""Definindo as condições de contorno"""

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
bc = [fem.dirichletbc(PETSc.ScalarType((0.0, 0.0)), left_dofs, V)]

ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag)

"""Tensores de deformação e tensão"""
v = ufl.TestFunction(V)
u = ufl.TrialFunction(V)

def I(u):
    d = len(u)
    return ufl.variable(ufl.Identity(d))


# Elasticity parameters
E = PETSc.ScalarType(1.0e4)
nu = PETSc.ScalarType(0.3)
mu = fem.Constant(domain, E/(2*(1 + nu)))
lmbda = fem.Constant(domain, E*nu/((1 + nu)*(1 - 2*nu)))



def epsilon(u):
    return ufl.sym(ufl.grad(u) + ufl.grad(u).T) 
def sigma(u):
    return lmbda * ufl.tr(epsilon(u)) * I(u) + 2*mu*epsilon(u)


"""Carregamento e tração"""
loading = fem.Constant(domain, PETSc.ScalarType((0, 0)))
Traction = fem.Constant(domain, PETSc.ScalarType((0, 0)))

#Defining the body force term
q = fem.Constant(domain, PETSc.ScalarType(0.0))

"""Formulação variacional"""
a = ufl.inner(ufl.grad(v), sigma(u))*ufl.dx
L= ufl.inner(v, loading)*ufl.dx - ufl.inner(v, Traction)*ds(2) 


#Forma variacional
b     = ufl.inner(v,u)*ufl.dx + dt*ufl.inner(ufl.grad(v),ufl.grad(u))*kappa*ufl.dx
c     = dt*ufl.inner(v,q)*ufl.dx + ufl.inner(v,u_init)*ufl.dx



problem = fem.petsc.LinearProblem(a, L, bcs = bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
uh = problem.solve()


file = XDMFFile(MPI.COMM_WORLD, "resultados/acoplamento_termo_elasticity.xdmf", "w")
file.write_mesh(domain)
file.write_function(uh)