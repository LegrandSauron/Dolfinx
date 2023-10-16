import numpy as np
import ufl
from dolfinx.io import XDMFFile
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, nls


domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([2,1])],
                  [40,40], cell_type=mesh.CellType.triangle)


x = ufl.SpatialCoordinate(domain)

"""espaços de funcoes"""

# Define function space, scalar
U2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)  # Displacent
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)  # temperature

# DOFs
TH = ufl.MixedElement([U2, P1])
V = fem.FunctionSpace(domain, TH)  # Total space for all DOFs

(u_test, T_test) = ufl.TestFunctions(V)  # Test function

# Define actual functions with the required DOFs
w = ufl.TrialFunction(V)
(u, Temperatura) = ufl.split(w)  # current DOFs

"""time parameters"""
t_init= 0.0
steps=100
Tempo_final= 1
dt= fem.Constant(domain,Tempo_final/steps)


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


def fixed_displacement_expression(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def incremented_displacement_expression(x):
    return np.full(x.shape[1], 2.0 ) #1.0e-02



V0, submap1 = V.sub(0).collapse()

fixed_displacement = fem.Function(V0)
fixed_displacement.interpolate(fixed_displacement_expression)
left_dof = fem.locate_dofs_topological((V.sub(0),V0), fdim, facet_tag.find(1))
bc1 = fem.dirichletbc(fixed_displacement, left_dof, V.sub(0))

VS, submap2 = V.sub(1).collapse()
fixed_displacement = fem.Function(VS)
fixed_displacement.interpolate(incremented_displacement_expression)
left_dof = fem.locate_dofs_topological((V.sub(1),VS), fdim, facet_tag.find(1))
bc2 = fem.dirichletbc(fixed_displacement, left_dof, V.sub(1))

bc= [bc1,bc2]

"""Solucao inicial"""
T_sf, submap3 = V.sub(1).collapse()
T_init = fem.Function(T_sf)
def initial_condition(x, a=PETSc.ScalarType(0.0)):
    return np.exp(a*(x[0])+ a*x[1])

T_init.interpolate(initial_condition)

#Hbath = 2.0

#ERROR
#u.sub(0).interpolate(lambda x: np.full((x.shape[1],), Hbath))
#u.x.scatter_forward()

ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag)

""" Elasticity parameters"""
E = PETSc.ScalarType(1.0e4)
nu = PETSc.ScalarType(0.3)
mu = fem.Constant(domain, E/(2*(1 + nu)))
lmbda =  E*nu/((1 + nu)*(1 - 2*nu))

#temperature parameters
alpha = 2.31e-5  # thermal expansion coefficient
kappa = alpha*(2*mu + 3*lmbda)

k = fem.Constant(domain,237e-6)  # thermal conductivity

"""Carregamento, tração, calor gerado"""
#Carregamento de tracao e distribuido
loading = fem.Constant(domain, PETSc.ScalarType((0, 0)))
Traction = fem.Constant(domain, PETSc.ScalarType((0, 0)))

#calor gerado
q = fem.Constant(domain, PETSc.ScalarType(0.0))

"""Tensores de deformação e tensão"""
def I(u):
    d = len(u)
    return ufl.variable(ufl.Identity(d)) 

def epsilon(u):
    return ufl.sym(ufl.grad(u) + ufl.grad(u).T)
def sigma(u,temp):
    return lmbda * ufl.tr(epsilon(u)) * I(u) + 2*mu*epsilon(u) - alpha*temp* ufl.Identity(2)



#Salvando os dados 
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "resultados/acopla.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    #xdmf.write_function(w.sub(1))
     
"""Formulação variacional"""
#Formulacao variacinal elasticidade
a = ufl.inner(ufl.grad(u_test), sigma(u,Temperatura))*ufl.dx + ufl.inner(u_test,u)*ufl.dx + dt*ufl.inner(ufl.grad(T_test),ufl.grad(Temperatura))*k*ufl.dx

L = ufl.inner(u_test, loading)*ufl.dx - ufl.inner(u_test, Traction)*ds(2) +  dt*ufl.inner(T_test,q)*ufl.dx + ufl.inner(T_test,T_init)*ufl.dx


"""Solucao"""
problem = fem.petsc.LinearProblem(a, L, bcs = bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})

for i in range(steps):
    t_init += dt
    uh 	  = problem.solve()
    o, oo = uh.split()
    T_init.interpolate(oo)
    xdmf.write_function(uh.sub(0),t_init)