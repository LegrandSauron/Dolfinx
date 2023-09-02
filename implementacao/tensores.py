from dolfinx import fem
import dolfinx
import ufl
from ufl import ds, dx, grad, inner, dot
from petsc4py.PETSc import ScalarType
from dolfinx.io import XDMFFile
from dolfinx import fem, mesh, log, nls
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc 
from petsc4py.PETSc import*
import petsc4py

# Scaled variable
carregamento= 50000
E = 78e6
poisson = 0.3
lambda_ = E*poisson / ((1+poisson)*(1-2*poisson))
G= E / 2*(1+poisson)



#Criando ou importando a geometria e realizando a discretização.
domain= mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([2,1])],
                  [20,20], cell_type=mesh.CellType.triangle)

#Importação da geometria e das condições de contorno.
#malha, cell_tags, facet_tags = gmshio.read_from_msh("malha_ece.msh", MPI.COMM_SELF,0, gdim=2)


#Realizando a aproximação do dominio por um espaço de funcao V  
V = fem.VectorFunctionSpace(domain, ("CG", 1))


#Definindo as condições de contorno
def clamped_boundary(x):
    return np.isclose(x[0], 0)


fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
u_D = np.array([0,0], dtype=ScalarType)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
Constante = fem.Constant(domain, ScalarType((0, 0)))

#Espaço de integração no contorno da geometria 
#ds = ufl.Measure("ds", domain=domain, subdomain_data= boundary_facets)
#dx = ufl.Measure("dx", domain=domain)


#Aproximação das funções teste e funcao incognita
v = ufl.TestFunction(V)
uh = fem.Function(V)
#uh  = ufl.TrialFunction(V)


#construindo o tensor de deformação , Cauchy-green-esquerdo
Identidade = (ufl.Identity(len(uh))) 
T_deform =(Identidade + ufl.grad(uh)) #Tensor de deformação
T_deform =ufl.as_tensor(T_deform)

Cauchy_g_Left= ufl.dot(T_deform,T_deform.T  )  #Cauchy-green-esquerdo

N, M = Cauchy_g_Left.ufl_shape



def Hencky_Strain(Cauchy_G_E):
    return ufl.as_tensor([[(1/2)*(ufl.ln(Cauchy_G_E[i,j])) for i in range(N)] for j in range(M)])
   


#Definindo o tensor de tensões T
def T_tensao(Strain): 
    return  (1/ufl.det(T_deform))*(2.0 * G * Strain + lambda_ * ufl.tr(Strain) * Identidade )


#funcao carregamento
f = fem.Constant(domain, ScalarType((0,carregamento )))

#formulação fraca a partir do funcional bilinear
a = ufl.inner(T_tensao(Hencky_Strain(Cauchy_g_Left)),ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ds

bilinear_form = fem.form(a)

J= ufl.derivative(a)

b = fem.petsc.create_vector(bilinear_form)
A = fem.petsc.assemble_matrix(bilinear_form, bcs=[bc])