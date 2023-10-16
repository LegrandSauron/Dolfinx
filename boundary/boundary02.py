
"""Ok"""

from mpi4py import MPI
from dolfinx import fem, io, nls, log, mesh, plot
import numpy as np
import pyvista
import ufl
from petsc4py.PETSc import ScalarType

L= 4
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([L,1])],
                  [20,20], cell_type=mesh.CellType.triangle)

"""parametros"""
T0 = fem.Constant(domain,293.)
DThole = fem.Constant(domain,10.)
E = 70e3
nu = 0.3
lmbda = fem.Constant(domain,E*nu/((1+nu)*(1-2*nu)))
mu = fem.Constant(domain,E/2/(1+nu))
rho = 2700.0    # density
alpha = 20.31e-5  # thermal expansion coefficient
kappa = alpha*(2*mu + 3*lmbda)
cV = fem.Constant(domain,910e-6)*rho  # specific heat per unit volume at constant strain
k = fem.Constant(domain,237e-6)  # thermal conductivity

"""Definindo as condições de contorno"""

boundaries = [
    (1, lambda x: np.isclose(x[0], 0)),
    (2, lambda x: np.isclose(x[0], L )),
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


"""Space functions"""
P1 = ufl.FiniteElement('Lagrange', domain.ufl_cell(), 1)
U1= ufl.VectorElement("Lagrange",domain.ufl_cell(),1)
element = ufl.MixedElement([U1, P1])

V = fem.FunctionSpace(domain, element)

U_ = ufl.TestFunction(V)
(u_, T_) = ufl.split(U_)

DU = ufl.TrialFunction(V)
(u, T) = ufl.split(DU)





"""Implemetando condições de contorno"""
V0, submap0 = V.sub(0).collapse()
V1,submap1= V.sub(1).collapse()

u_D0 = fem.Function(V0)
u_D0.x.array[:] = 0.0

u_D1 = fem.Function(V1)
u_D1.x.array[:] = 300.0


boundary_dofs_b0 = fem.locate_dofs_topological((V.sub(0), V0), fdim, facet_tag.find(1))
bc0 = fem.dirichletbc(u_D0, boundary_dofs_b0, V.sub(0))


boundary_dofs_b3 = fem.locate_dofs_topological((V.sub(0), V0), fdim, facet_tag.find(2))
bc3 = fem.dirichletbc(u_D0, boundary_dofs_b3, V.sub(0))



boundary_dofs_b1 = fem.locate_dofs_topological((V.sub(1), V1), fdim, facet_tag.find(1))
bc1 = fem.dirichletbc(u_D1, boundary_dofs_b1, V.sub(1))

boundary_dofs_b2 = fem.locate_dofs_topological((V.sub(1), V1), fdim, facet_tag.find(2))
bc2 = fem.dirichletbc(u_D0, boundary_dofs_b2, V.sub(1))


bc= [bc0,bc1,bc2]



def eps(v):
    return ufl.sym(ufl.grad(v))


def sigma(v, Theta):
    return (lmbda*ufl.tr(eps(v)) - kappa*Theta)*ufl.Identity(2) + 2*mu*eps(v)

s     = fem.Constant(domain, ScalarType(0.0))
q = fem.Constant(domain, ScalarType(0.0))

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

mech_form = ufl.inner(sigma(u, T), eps(u_))*ufl.dx

therm_form = ufl.inner(ufl.grad(T),ufl.grad(T_))*k*ufl.dx + ufl.inner(q,T_)*k*ds(2) + ufl.inner(s,T_)*ufl.dx


form = mech_form + therm_form

"""Solucao"""
problem = fem.petsc.LinearProblem(ufl.lhs(form), ufl.rhs(form),bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})

uh = problem.solve()

sigma_h, u_h = uh.sub(0).collapse(), uh.sub(1).collapse()



from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "resultados/acopla.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(sigma_h)


    

   
    