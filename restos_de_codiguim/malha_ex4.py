import gmsh
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI


gmsh.initialize()
gmsh.open("exemplo_4.msh")
gmsh.model.add("DFG 3D")



gmsh.model.occ.synchronize()


gmshio.extract_geometry(gmsh.model) #Returns: The mesh geometry as an array of shape (num_nodes, 3).


#mesh, cell_tags, facet_tags = gmshio.read_from_msh(gmsh.model,MPI.COMM_WORLD,0,2)


model_rank = 0
comm= MPI.Comm
gdim = 2
shape = "triangle"
degree = 1

#x = gmshio.extract_geometry()

#cell = ufl.Cell("triangle", geometric_dimension=gdim)
#domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))




#ufl_domain = gmshio.ufl_mesh(cell, 1)

#mesh.create_mesh(MPI.COMM_SELF, cell, x, ufl_domain)
model_rank = 0
mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, model_rank)

# This functio