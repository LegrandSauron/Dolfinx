import dolfinx as dfx
import numpy as np
import ufl
from mpi4py import MPI

msh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 1, 1)

sz = 3
elem = ufl.FiniteElement("CG", msh.ufl_cell(), 1)
elem_mixed = ufl.MixedElement([elem] * sz)
V_vec = dfx.fem.VectorFunctionSpace(msh, ("CG", 1), dim=sz)
V_mix = dfx.fem.FunctionSpace(msh, elem_mixed)

v_vec = dfx.fem.Function(V_vec)
v_mix = dfx.fem.Function(V_mix)

for i in range(sz):
    v_vec.sub(i).interpolate(lambda x: np.full_like(x[0], i))
    v_mix.sub(i).interpolate(lambda x: np.full_like(x[0], i))

print("vector:", v_vec.x.array)
print("mixed:", v_mix.x.array)

V_mix_0, map_0 = V_mix.sub(0).collapse()
offset = V_mix_0.dofmap.index_map.size_local * V_mix_0.dofmap.index_map_bs
print("mixed component 0 (incorrect):", v_mix.x.array[:offset])
print("mixed component 0 (correct):", v_mix.x.array[map_0])