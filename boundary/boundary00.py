from mpi4py import MPI
from dolfinx import fem, io, nls, log, mesh, plot
import numpy as np
import ufl
from petsc4py.PETSc import ScalarType


domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([2,1])],
                  [40,40], cell_type=mesh.CellType.triangle)

"""parametros"""
T0 = fem.Constant(domain,293.)
DThole = fem.Constant(domain,10.)
E = 70e3
nu = 0.3
lmbda = fem.Constant(domain,E*nu/((1+nu)*(1-2*nu)))
mu = fem.Constant(domain,E/2/(1+nu))
rho = 2700.0    # density
alpha = 2.31e-5  # thermal expansion coefficient
kappa = alpha*(2*mu + 3*lmbda)
cV = fem.Constant(domain,910e-6)*rho  # specific heat per unit volume at constant strain
k = fem.Constant(domain,237e-6)  # thermal conductivity


"""time parameters"""
tempo_init= 0.0
steps=100
tempo_final= 100
dt= fem.Constant(domain,tempo_final/steps)



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

"""Space functions"""
P1 = ufl.FiniteElement('Lagrange', domain.ufl_cell(), 1)
U1= ufl.VectorElement("Lagrange",domain.ufl_cell(),1)
element = ufl.MixedElement([U1, P1])

V = fem.FunctionSpace(domain, element)

U_ = ufl.TestFunction(V)
(u_, Theta_) = ufl.split(U_)
dU = ufl.TrialFunction(V)
(du, dTheta) = ufl.split(dU)
Uold = fem.Function(V)
(uold, Thetaold) = ufl.split(Uold)



"""Implemetando condições de contorno"""
V0, submap = V.sub(1).collapse()
V1,submap1= V.sub(0).collapse()

u_D1 = fem.Function(V1)
u_D1.x.array[:] = 0.0
u_D2 = fem.Function(V0)
u_D2.x.array[:] = 0.0

boundary_dofs_b = fem.locate_dofs_topological((V.sub(0), V1), fdim, facet_tag.find(1))
bc1 = fem.dirichletbc(u_D1, boundary_dofs_b, V.sub(0))

boundary_dofs_b = fem.locate_dofs_topological((V.sub(1), V0), fdim, facet_tag.find(1))
bc2 = fem.dirichletbc(u_D1, boundary_dofs_b, V.sub(1))

bc= [bc1,bc2]

"""Implementando a condição inicial"""
def y0_init(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = 1.0
    return values

def y1_init(x):
    values = np.zeros((1, x.shape[1]))
    values[:] = 2.0
    return values


def fixed_displacement_expression(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def incremented_displacement_expression(x):
    return np.full(x.shape[1], 1.0e-02)

def fixed_director_expression(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1])))


#Uold.sub(1).interpolate(y0_init)
#Uold.sub(0).interpolate(fixed_displacement_expression)

def eps(v):
    return ufl.sym(ufl.grad(v))


def sigma(v, Theta):
    return (lmbda*ufl.tr(eps(v)) - kappa*Theta)*ufl.Identity(2) + 2*mu*eps(v)


dt = fem.Constant(domain, 0.)
mech_form = ufl.inner(sigma(du, dTheta), eps(u_))*ufl.dx
therm_form = (cV*(dTheta-Thetaold)/dt*Theta_ +
              kappa*T0*ufl.tr(eps(du-uold))/dt*Theta_ +
              ufl.dot(k*ufl.grad(dTheta), ufl.grad(Theta_)))*ufl.dx

form = mech_form + therm_form





"""Solucao"""
problem = fem.petsc.LinearProblem(ufl.lhs(form), ufl.rhs(form), bcs = bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})


#Salvando os dados 
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "resultados/acopla.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)



L = 1.
R = 0.1    
Nincr = 100
t = np.logspace(1, 4, Nincr+1)
Nx = 100
x = np.linspace(R, L, Nx)
T_res = np.zeros((Nx, Nincr+1))

for (i, dti) in enumerate(np.diff(t)):
    print("Increment " + str(i+1))
    dt +=(dti)
    uh = problem.solve()
    Uold.sub(1).interpolate(uh.sub(1))
    xdmf.write_function(Uold.sub(1),dt)
    