from mpi4py import MPI
from dolfinx import fem, io, nls, log, mesh, plot
import numpy as np
import ufl
from petsc4py.PETSc import ScalarType

L = 1
domain = mesh.create_interval(MPI.COMM_WORLD, 100, [0, L])
P1 = ufl.FiniteElement('CG', domain.ufl_cell(), 1)
element = ufl.MixedElement([P1, P1])
V = fem.FunctionSpace(domain, element)

# test functions
v1, v2 = ufl.TestFunctions(V)
# fields to solve for
u = fem.Function(V)
u1, u2 = u.split()
u.name = "u"
# function determining if a node is on the tray top
def on_top_boundary(x):
    return(np.isclose(x[0], L))

V0, submap = V.sub(0).collapse()

# determine boundary DOFs
boundary_dofs = fem.locate_dofs_geometrical((V.sub(0),V0), on_top_boundary)


# apply dirichlet BC to boundary DOFs
C_0 = 5
bc = fem.dirichletbc(ScalarType(C_0), boundary_dofs[0], V.sub(0))

u.x.array[bc.dof_indices()[0]] = bc.value.value

with io.XDMFFile(domain.comm, "u.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u.sub(0))
    xdmf.write_function(u.sub(1))