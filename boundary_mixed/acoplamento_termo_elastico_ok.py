
from mpi4py import MPI
from dolfinx import fem, io, nls, log, mesh, plot
import numpy as np
import pyvista
from ufl import VectorElement,FiniteElement,MixedElement,TestFunction,TrialFunction,split,grad,tr,Identity,inner,dot
from petsc4py.PETSc import ScalarType
import ufl

L= 1
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([L,1])],
                  [20,20], cell_type=mesh.CellType.triangle)

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

"""Parameters"""
T0 = fem.Constant(domain,293.)
DThole = fem.Constant(domain,10.)
E = 70e3
nu = 0.3
lmbda = fem.Constant(domain,E*nu/((1+nu)*(1-2*nu)))
mu = fem.Constant(domain,E/2/(1+nu))
rho = 2700.0    # density
alpha = 20.31e-6  # thermal expansion coefficient
kappa = alpha*(2*mu + 3*lmbda)
cV = fem.Constant(domain,910e-6)*rho  # specific heat per unit volume at constant strain
k = fem.Constant(domain,237e-6)  # thermal conductivity


"""time parameters"""
tempo_init= 0.0
steps=100
tempo_final= 1000
dt= fem.Constant(domain,tempo_final/steps)


"""Function Space"""
Vue = VectorElement('Lagrange', domain.ufl_cell(), 1) # displacement finite element
Vte = FiniteElement('Lagrange', domain.ufl_cell(), 1)
Element= MixedElement([Vue, Vte]) # temperature finite element
V = fem.FunctionSpace(domain, Element)

#Extraindo os sub espaços do elemento misto e os mapas contendo os graus de liberdade 
num_subs = V.num_sub_spaces
spaces = []
maps = []
for i in range(num_subs):
    space_i, map_i = V.sub(i).collapse()
    spaces.append(space_i)
    maps.append(map_i)

(u_, Theta_) = ufl.TestFunctions(V)

W = TrialFunction(V)

(du, dTheta) = split(W)

Wold = fem.Function(V)
Wold.x.array[map_i[1]]= 0.0 #Errado?
Wold.x.array[map_i[0]]= 0.0 #errado ?

(uold, Thetaold) = Wold.sub(0), Wold.sub(1)


"""Bondary condition """
V0, submap0 = V.sub(0).collapse()
V1,submap1= V.sub(1).collapse()
u_D0 = fem.Function(V0)
u_D0.x.array[:] = 0.0 #Teste usando np.arange para implementar a matriz de valores da condição de contorno

u_D1 = fem.Function(V1)
u_D1.x.array[:] = DThole

u_D2= fem.Function(V0)
#u_D2.x.array[]
print(len(u_D2.x.array))
#Engastes

boundary_dofs_b0 = fem.locate_dofs_topological((V.sub(0), V0), fdim, facet_tag.find(1))
bc0 = fem.dirichletbc(u_D0, boundary_dofs_b0, V.sub(0))

#Temperaturas 
boundary_dofs_b1 = fem.locate_dofs_topological((V.sub(1), V1), fdim, facet_tag.find(1))
bc1 = fem.dirichletbc(u_D1, boundary_dofs_b1, V.sub(1))
boundary_dofs_b2 = fem.locate_dofs_topological((V.sub(1), V1), fdim, facet_tag.find(2))
bc2 = fem.dirichletbc(u_D0, boundary_dofs_b2, V.sub(1))

#Engaste
boundary_dofs_b3 = fem.locate_dofs_topological((V.sub(0), V0), fdim, facet_tag.find(3))
bc3 = fem.dirichletbc(u_D0, boundary_dofs_b3, V.sub(0))
boundary_dofs_b4 = fem.locate_dofs_topological((V.sub(0), V0), fdim, facet_tag.find(4))
bc4 = fem.dirichletbc(u_D0, boundary_dofs_b4, V.sub(0))

bc= [bc0,bc1]

"""Tensores"""
def eps(v):
    return ufl.sym(grad(v))

def sigma(v, Theta):
    return (lmbda*tr(eps(v)) - kappa*Theta)*Identity(2) + 2*mu*eps(v)

"""Formulação variacional"""
mech_form = inner(sigma(du, dTheta), eps(u_))*ufl.dx

#therm_form = (cV*(dTheta-Thetaold)/dt*Theta_ +kappa*T0*tr(eps(du-uold))/dt*Theta_ + dot(k*grad(dTheta), grad(Theta_)))*ufl.dx
q= fem.Constant(domain,ScalarType(0.0))
#Forma variacional
therm_form = (cV*(dTheta-Thetaold)/dt*Theta_ +
              kappa*T0*ufl.tr(eps(du-uold))/dt*Theta_ +
              ufl.dot(k*ufl.grad(dTheta), ufl.grad(Theta_)))*ufl.dx + dt*ufl.inner(Theta_,q)*ufl.ds(3) + dt*ufl.inner(Theta_,q)*ufl.ds(4)

form = mech_form + therm_form

problem = fem.petsc.LinearProblem(ufl.lhs(form), ufl.rhs(form),bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "resultados/tcm_2D.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

for i in range(steps):
    tempo_init += dt
    uh= problem.solve()
    xdmf.write_function(uh.sub(1),tempo_init)
    a= uh.sub(1)
    Thetaold.x.array[:]= a.x.array    
    