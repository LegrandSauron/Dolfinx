import dolfinx
import ufl
from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace, Constant
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import assemble_scalar
from dolfinx.fem import locate_dofs_topological
from dolfinx.io import VTKFile
from dolfinx.mesh import create_interval
from mpi4py import MPI
import numpy as np
import scipy.sparse as sp
from petsc4py.PETSc import ScalarType

# Define a função que representa a solução da equação de condução de calor
def solve_heat_conduction(L, T, Nx, Nt, k):
    # Crie uma malha unidimensional
    mesh = create_interval(MPI.COMM_WORLD,20,np.array([0,L]) )

    # Defina o espaço de funções para a temperatura
    V = FunctionSpace(mesh, ("Lagrange", 1))

    # Defina a função de teste e a função de teste anterior no espaço de funções
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # Crie a função que representa a temperatura
    u0 = Function(V)
    u1 = Function(V)

    # Condição inicial
    u_0 = Constant(mesh, ScalarType(0.0))
    u0.interpolate(u_0)

    # Parâmetros de tempo
    dt = T / Nt
    t = 0.0

    # Loop de tempo
    for n in range(Nt):
        # Formulando o problema variacional
        F = (u - u0) * v * ufl.dx + k * dt * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - (u1 - u0) * v * ufl.dx

        # Resolver o problema variacional
        a, L = ufl.split(ufl.lhs(F)), ufl.split(ufl.rhs(F))
        A = assemble_scalar(a)
        b = assemble_scalar(L)

        # Aplicar condição de contorno de Dirichlet em ambos os extremos da barra
        bc = dolfinx.DirichletBC(u0, locate_dofs_topological(V, mesh.topology.dim - 1, lambda x: np.isclose(x[0], 0.0)))
        bc.apply(A, b)

        # Resolver o sistema linear
        solver = dolfinx.LUSolver(MPI.COMM_WORLD)
        solver.set_operator(A)
        solver.solve(u1.vector, b)

        # Atualizar as soluções anteriores
        u0.vector.set_local(u1.vector.get_local())

        # Avançar no tempo
        t += dt

    return u1

if __name__ == "__main__":
    L = 1.0  # Comprimento da barra
    T = 1.0  # Tempo final
    Nx = 100  # Número de elementos da malha
    Nt = 100  # Número de passos de tempo
    k = 0.01  # Coeficiente de difusão térmica

    # Resolver a equação de condução de calor
    solution = solve_heat_conduction(L, T, Nx, Nt, k)

    # Salvar a solução em um arquivo VTK
    vtkfile = VTKFile(MPI.COMM_WORLD, "heat_conduction_solution.pvd", "w")
    vtkfile.write(solution, T)
