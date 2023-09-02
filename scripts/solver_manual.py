from dolfinx import fem
import ufl
from petsc4py.PETSc import ScalarType
from dolfinx import fem,log, nls ,mesh
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc




class Problem(fem.petsc.NonlinearProblem):
    def __init__(self, J, J_pc, F, bcs):
        self.bilinear_form = J
        self.preconditioner_form = J_pc
        self.linear_form = F
        self.bcs = bcs
        fem.petsc.NonlinearProblem.__init__(self)

    def F(self, b, x):
        pass

    def J(self, A, x):
        pass

    def form(self, A, P, b, x):
        fem.assemble(self.linear_form, tensor=b)
        fem.assemble(self.bilinear_form, tensor=A)
        fem.assemble(self.preconditioner_form, tensor=P)
        for bc in self.bcs:
            bc.apply(b, x)
            bc.apply(A)
            bc.apply(P)


class CustomSolver(nls.petsc.NewtonSolver):
    def __init__(self):
        nls.petsc.NewtonSolver.__init__(self, mesh.mpi_comm(),
                              PETSc.PETScKrylovSolver(), PETSc.PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operators(A, P)

        PETSc.PETScOptions.set("ksp_type", "minres")
        PETSc.PETScOptions.set("pc_type", "hypre")
        PETSc.PETScOptions.set("ksp_view")

        self.linear_solver().set_from_options()

