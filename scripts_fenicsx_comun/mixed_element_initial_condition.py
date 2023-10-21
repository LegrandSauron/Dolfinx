import dolfinx
from mpi4py import MPI
import dolfinx.io
import ufl
import numpy as np

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
el = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
mel = ufl.MixedElement([el, el])
V = dolfinx.fem.FunctionSpace(mesh, mel)
num_subs = V.num_sub_spaces
spaces = []
maps = []
for i in range(num_subs):
    space_i, map_i = V.sub(i).collapse()
    spaces.append(space_i)
    maps.append(map_i)


u = dolfinx.fem.Function(V)
u0 = dolfinx.fem.Function(spaces[0])
u1 = dolfinx.fem.Function(spaces[1])
u0.x.array[:] = np.arange(len(maps[0]))
u1.x.array[:] = np.arange(len(maps[1]))[::-1]
u.x.array[maps[0]] = u0.x.array
u.x.array[maps[1]] = u1.x.array
u.x.scatter_forward()
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "splitted_space.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u.sub(0))
    xdmf.write_function(u.sub(1))