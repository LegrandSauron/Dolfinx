from dolfinx import fem
import ufl
from dolfinx import fem, mesh, nls, log
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc 
from petsc4py.PETSc import*
 
#criando a geometria e o numero de elementos
domain= mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0,0.0]), np.array([1.0, 1.0])],
                  [20,20], cell_type=mesh.CellType.triangle)

x = ufl.SpatialCoordinate(domain)

#definindo o espaço de funções 
Va=fem.VectorFunctionSpace(domain, ("CG", 1))

#tensor cauchy green esquerdo
hencky_strain = Hencky_Strain(Va)

if __name__ == "__main__":
    pass

        



