import meshio
import h5py
import numpy as np
from mpi4py import MPI

import gmsh
from dolfinx import fem, io, mesh, plot
from petsc4py.PETSc import ScalarType
from ufl import ds, dx, grad, inner 
from dolfinx.io import XDMFFile, gmshio
import ufl

gmsh.initialize()
model=gmsh.open("exemplo_6_surf.msh")
malha= meshio.read("exemplo_6.xdmf")
#dominio = meshio.Mesh(malha.points, malha.cells)

mesh_name = "exemplo_6"
msh = meshio.read(mesh_name+".msh")

"""tri_data = msh.cell_data_dict["gmsh:physical"]["triangle"]
meshio.write(mesh_name+"_surf.xdmf",
    meshio.Mesh(points=msh.points,
        cells={"triangle": msh.cells_dict["triangle"]},
        cell_data={"surf_marker": [tri_data]}
    )
)
"""
points= msh.points
# Converter para objeto dolfinx.mesh.Mesh
#meshe = gmshio.read_from_msh("exemplo_6.msh",MPI.COMM_WORLD,0,3)
#a,b,c =gmshio.model_to_mesh(model,MPI.COMM_WORLD,0,3)
#meshe = mesh.create_mesh(MPI.COMM_WORLD,points,tri_data)

ufl.as_cell(model)
#meshio.write("exemplo_6.xdmf", malha)


#V= fem.FunctionSpace(meshe,("CG",1))