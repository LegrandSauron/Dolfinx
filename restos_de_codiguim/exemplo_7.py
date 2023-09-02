import warnings
warnings.filterwarnings("ignore")
import gmsh
gmsh.initialize()
from dolfinx.io import gmshio


gmsh.open("exemplo_7_2d.msh")


x = gmshio.extract_geometry(gmsh.model)
topologies = gmshio.extract_topology_and_markers(gmsh.model)


# The mesh geometry is a (number of points, 3) array of all the coordinates of the nodes in the mesh. The topologies is a dictionary, where the key is the unique [GMSH cell identifier](http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format). For each cell type, there is a sub-dictionary containing the mesh topology, an array (number_of_cells, number_of_nodes_per_cell) array containing an integer referring to a row (coordinate) in the mesh geometry, and a 1D array (cell_data) with mesh markers for each cell in the topology.
#
# As an MSH-file can contain meshes for multiple topological dimensions (0=vertices, 1=lines, 2=surfaces, 3=volumes), we have to determine which of the cells has to highest topological dimension. We do this with the following snippet

# +
import numpy

# Get information about each cell type from the msh files
num_cell_types = len(topologies.keys())
cell_information = {}
cell_dimensions = numpy.zeros(num_cell_types, dtype=numpy.int32)
for i, element in enumerate(topologies.keys()):
    properties = gmsh.model.mesh.getElementProperties(element)
    name, dim, order, num_nodes, local_coords, _ = properties
    cell_information[i] = {"id": element, "dim": dim,
                           "num_nodes": num_nodes}
    cell_dimensions[i] = dim


gmsh.finalize()

# Sort elements by ascending dimension
perm_sort = numpy.argsort(cell_dimensions)
# -

# We extract the topology of the cell with the highest topological dimension from `topologies`, and create the corresponding `ufl.domain.Mesh` for the given celltype

cell_id = cell_information[perm_sort[-1]]["id"]
cells = numpy.asarray(topologies[cell_id]["topology"], dtype=numpy.int64)
ufl_domain = gmshio.ufl_mesh(cell_id, 2)

# As the GMSH model has the cell topology ordered as specified in the  [MSH format](http://gmsh.info//doc/texinfo/gmsh.html#Node-ordering),
# we have to permute the topology to the [FIAT format](https://github.com/FEniCS/dolfinx/blob/e7f0a504e6ff538ad9992d8be73f74f53b630d11/cpp/dolfinx/io/cells.h#L16-L77). The permuation is done using the `perm_gmsh` function from DOLFINx.

from dolfinx.cpp.io import perm_gmsh
from dolfinx.cpp.mesh import to_type

num_nodes = cell_information[perm_sort[-1]]["num_nodes"]
gmsh_cell_perm = perm_gmsh(to_type(str(ufl_domain.ufl_cell())), num_nodes)
cells = cells[:, gmsh_cell_perm]


# The final step is to create the mesh from the topology and geometry (one mesh on each process).

from dolfinx.mesh import create_mesh
from mpi4py import MPI
mesh = create_mesh(MPI.COMM_SELF, cells, x, ufl_domain)

from dolfinx.io import XDMFFile
with XDMFFile(MPI.COMM_WORLD, "mesh_outoter_1.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
