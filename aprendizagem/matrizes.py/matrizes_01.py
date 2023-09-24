
# Fenics-related packages
import pyvista
import ufl
from dolfinx import fem, mesh
import numpy as np
import math
# Plotting packages
import matplotlib.pyplot as plt
# Current time package

from dolfinx.io import gmshio
from mpi4py import MPI
from petsc4py.PETSc import ScalarType



#criando a geometria e o numero de elementos
domain= mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0,0.0]), np.array([1.0, 1.0])],
                  [16,16], cell_type=mesh.CellType.triangle)

V= fem.FunctionSpace(domain, ("CG", 1))

boundaries = [
    (1, lambda x: np.isclose(x[0], 0)),
    (2, lambda x: np.isclose(x[0], 4)),
    (3, lambda x: np.isclose(x[1], 0)),
    (4, lambda x: np.isclose(x[1], 4))
]
  
facet_indices, facet_markers = [], [] #matriz para criação dos indices de cada face e a face em si
fdim = domain.topology.dim - 1

for (marker, locator) in boundaries:  #função percorre a lista boudaries, onde (marker = indices (1,2,3), locator= faces
    facets = mesh.locate_entities(domain, fdim, locator) #coloca as faces (locator) dentro da variavel faces
    facet_indices.append(facets) #
    facet_markers.append(np.full_like(facets, marker))

facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)

facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

#COndções de contorno
from petsc4py.PETSc import ScalarType
u_D = np.array((0,) * domain.geometry.dim, dtype=ScalarType)
 
dofs_1 = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bcs_1=fem.dirichletbc(0.0,dofs=dofs_1,V=V)

dofs_2 = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(2))
bcs_2=fem.dirichletbc(0.0,dofs=dofs_2,V=V)


bcs= [bcs_1,bcs_2]


"""Definindo os restante das condições para simulação de um problema...."""
x = ufl.SpatialCoordinate(domain)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

a = ufl.inner(ufl.grad(v), ufl.grad(u)) * ufl.dx 
f= 2*(ufl.pi**2)* ufl.sin(ufl.pi*x[0])*ufl.sin(2*ufl.pi*x[1])       #2π2sin(πx)sin(2πy) 
L = v*f *ufl.dx

problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

#carregamento
Q = fem.FunctionSpace(domain, ("CG", 2))
expr = fem.Expression(f, Q.element.interpolation_points())
pressure = fem.Function(Q)
pressure.interpolate(expr)
pressure.name = "Load"

uh.name = "deslocamento"

"""Vizualização dos resultados"""

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "resultados/matrizes_01.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
    xdmf.write_function(pressure)


from dolfinx.plot import create_vtk_mesh
import pyvista

load_plotter = pyvista.Plotter()
p_grid = pyvista.UnstructuredGrid(*create_vtk_mesh(Q))
p_grid.point_data["p"] = pressure.x.array.real
warped_p = p_grid.warp_by_scalar("p", factor=1)
warped_p.set_active_scalars("p")
load_plotter.add_mesh(warped_p, show_scalar_bar=True)
load_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    load_plotter.show()
else:
    load_plotter.screenshot("load.png")