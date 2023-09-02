from dolfinx import fem
import ufl
from petsc4py.PETSc import ScalarType
from dolfinx import fem,log, nls ,mesh
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc

# Scaled variable
carregamento= -20
E = 30e6
poisson = 0.3
lambda_ = E*poisson / ((1+poisson)*(1-2*poisson))
G= E / 2*(1+poisson)



domain= mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([2,1])],
                  [100,100], cell_type=mesh.CellType.triangle)

V = fem.VectorFunctionSpace(domain, ("CG", 1))
#condições de contorno
def clamped_boundary(x):
    return np.isclose(x[0], 0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
u_D = np.array([0,0], dtype=ScalarType)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)


#Import geometry by gmsh
#domain, cell_tags, facet_tags = gmshio.read_from_msh("malha_150_60_acopla.msh", MPI.COMM_WORLD,0, gdim=2)


#application of boundary conditions , crimping.
""" Faces para aplicação das condições de contorno
Physical Curve("esgaste", 16) = {14, 15, 3, 1};
Physical Curve("carregamento", 17) = {8};
Physical Surface("dominio", 18) = {1, 2, 3, 4};
"""

#u_D = np.array([0,0], dtype=ScalarType) 
#dofs_2 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(16))
#bc=fem.dirichletbc(u_D,dofs=dofs_2,V=V)


#Aproximação das funções teste e funcao incognita
v = ufl.TestFunction(V)
u= fem.Function(V)

#hencky's strain tensor
d = len(u)
I = (ufl.Identity(d)) 
F =( I + ufl.grad(u)) 
N_F , m_F = F.ufl_shape



#tensor de cauchy-Green left 
C= (F * F.T )

#determinando o Jacobiano
Jhc_e= ufl.det(F)
J= Jhc_e

#Hencky's Strain
N, M = C.ufl_shape
T_hencky=ufl.as_tensor(
    [[(1/2)* (ufl.ln(C[i,j])) for i in range(N)]
      for j in range(M)]
    )


def Henck_Strain(C):
    N, M = C.ufl_shape
    for i in range (N):
        for j in range(M):
            ufl.ln(C[i,j])
            


#for i in range(N):
  #  for j in range(M):

 #     print ("\n",I[i,j])



Jhc_e= ufl.det(F)

T_tension= (1/J)*(2.0 * G * T_hencky+ lambda_ * ufl.tr(T_hencky) * I )



#[0.5 * ln(({ A | A_{i_8, i_9} = sum_{i_{10}} (I + (grad(f)))[i_8, i_{10}] * ((I + (grad(f)))^T)[i_{10}, i_9]  })[0, 1]), 0.5 * ln(({ A | A_{i_8, i_9} = sum_{i_{10}} (I + (grad(f)))[i_8, i_{10}] * ((I + (grad(f)))^T)[i_{10}, i_9]  })[1, 1])]

print(ufl.ln(ufl.grad(u[0])[0]))

print(ufl.rank(ufl.grad(v)))