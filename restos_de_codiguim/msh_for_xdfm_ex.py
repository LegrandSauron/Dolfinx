import meshio
import numpy as np
from dolfinx import mesh





mesh_name = "exemplo_6"
msh = meshio.read(mesh_name+".msh")

tri_data = msh.cell_data_dict["gmsh:physical"]["triangle"]
meshio.write(mesh_name+"_surf.xdmf",
    meshio.Mesh(points=msh.points,
        cells={"triangle": msh.cells_dict["triangle"]},
        cell_data={"surf_marker": [tri_data]}
    )
)
#line_data = msh.cell_data_dict["gmsh:physical"]["line"]
#meshio.write(mesh_name+"_line.xdmf",meshio.Mesh(points=msh.points,cells={"line": msh.cells_dict["line"]},cell_data={"line_marker": [line_data]}))

points = msh.points

