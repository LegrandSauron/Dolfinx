from mpi4py import MPI
#from dolfinx.generation import UnitSquareMesh
from dolfinx.cpp.mesh import CellType
from dolfinx import fem ,mesh
from ufl import TestFunctions, TrialFunctions, VectorElement, FiniteElement, MixedElement, dot, grad, div, dx, lhs, rhs
import numpy as np

L_comprimento = 0.1
dominio= mesh.create_interval(MPI.COMM_WORLD,30,np.array([0,L_comprimento]) )

U_el = VectorElement("Lagrange", dominio.ufl_cell(), 1)
P_el = FiniteElement("Lagrange", dominio.ufl_cell(), 1)
W = fem.FunctionSpace(dominio, MixedElement([U_el, P_el]))

(ua_n2, pa_n2) = TrialFunctions(W) 
(qu, qp) = TestFunctions(W)
up = fem.Function(W)

U, U_to_W = W.sub(0).collapse()
P, P_to_W = W.sub(1).collapse()

ua_n1 = fem.Function(U)
pa_n1 = fem.Function(P)


F = (\
        dot(ua_n2-ua_n1+grad(pa_n2) , qu) + dot(pa_n2-pa_n1+div(ua_n2), qp))*dx
         

from petsc4py import PETSc         
problem = fem.petsc.LinearProblem(lhs(F),  rhs(F) , u=up)
problem.solve()
_ua, _pa = up.split()


ua_n1.x.array[:] = up.x.array[U_to_W]
ua_n1.x.scatter_forward()    

pa_n1.x.array[:] = up.x.array[P_to_W]
pa_n1.x.scatter_forward()



#Salvando os dados 
from dolfinx.io import XDMFFile
with XDMFFile(dominio.comm, "resultados/acopla.xdmf", "w") as xdmf:
    xdmf.write_mesh(dominio)
  
    #xdmf.write_function()