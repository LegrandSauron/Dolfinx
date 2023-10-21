import numpy as np
from petsc4py import PETSc
import ufl
from ufl import (MixedElement)
from dolfinx import (fem, io)
from dolfinx.io.gmshio import read_from_msh
from mpi4py import MPI

"""
External mesh loading
"""
domain, ct, ft = read_from_msh(
    filename="mesh.msh",
    comm=MPI.COMM_WORLD,
    rank=0, gdim=2
)
gdim = domain.topology.dim
fdim = gdim - 1
n = ufl.FacetNormal(
    domain=domain
)
surf_upside = 14
surf_dnside = 13
line_noslip = 12


"""
Spaces and Functions
"""
degree = 2
DG_elem = ufl.VectorElement(
    family="Discontinuous Lagrange",
    cell=domain.ufl_cell(),
    degree=degree
)
CG_elem = ufl.FiniteElement(
    family="Lagrange",
    cell=domain.ufl_cell(),
    degree=degree - 1
)
MixedSpace = fem.FunctionSpace(
    mesh=domain,
    element=MixedElement([DG_elem, CG_elem])
)
v, q = ufl.TestFunctions(
    function_space=MixedSpace
)
state = fem.Function(
    V=MixedSpace,
    name="Current state"
)

"""
Load and assign the initial condition
"""
# def init_h(x):
#     down_idx = x[1] > 0.4
#     values = np.full((x.shape[1],), 0.3)
#     values[down_idx] = 0.01
#     return values
#
#
# state.sub(1).interpolate(lambda x: init_h(x))

space1, map1 = state.sub(1).collapse()

bottom_cells = ct.find(13)
space1[map1[bottom_cells]] = np.full_like(map1[bottom_cells], 1, dtype=PETSc.ScalarType)
top_cells = ct.find(14)
space1[map1[top_cells]] = np.full_like(map1[top_cells], 0.1, dtype=PETSc.ScalarType)


with io.XDMFFile(comm=MPI.COMM_WORLD,
                 filename="state.xdmf",
                 file_mode="w",
                 encoding=io.XDMFFile.Encoding.HDF5) as xdmf:
    xdmf.write_mesh(mesh=domain)
    xdmf.write_function(u=state.sub(1).collapse())