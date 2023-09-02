import gmsh
gmsh.initialize()

gmsh.fltk()

gmsh.finalize()


from dolfinx.mesh import create_mesh
from mpi4py import MPI

from dolfinx.io import gmshio
mesh, cell_tags, facet_tags = gmshio.read_from_msh("malha_semi_pronta.msh", MPI.COMM_WORLD, 0, gdim=2)


# Output DOLFINx meshes to file
from dolfinx.io import XDMFFile

with XDMFFile(MPI.COMM_WORLD, "mesh_out.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(facet_tags)
    xdmf.write_meshtags(cell_tags)
# -