"""Criação de SUBDOMINIOS PARA DETERMINAÇÃO DE PROPRIEDADES DIFERENTES, EM DIFERENTES REGIOES DA GEOMETRIA."""
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx import fem
import numpy as np
from petsc4py import PETSc
import pyvista
import ufl
from petsc4py.PETSc import ScalarType

domain, cell_tags, facet_tags = gmshio.read_from_msh("malha_com_eletrodo_04.msh", MPI.COMM_WORLD,0, gdim=3)

"""
Obtenção das faces para aplicação das condições de contorno são dadas por : indices: 1=faces, 2=superficies, 3=volumes
1 14 "eletrodo_superior_l"
1 15 "gel_superior_l"
1 16 "gel_inferior_l"
1 17 "eletrodo_inferior_l"
1 18 "eletrodo_inferior_r"
1 19 "gel_inferior_r"
1 20 "gel_superior_r"
1 21 "eletrodo_superior_r"
2 26 "eletrodo_superior"
2 27 "eletrodo_inferior"
2 28 "gel"
"""

"""
Determinando materiais diferentes em uma geometria que possui entidades fisicas pre-definidas:
    -Cria-se um espaço de funções descontinuas para interpolação

    -Determina-se as faces, superficies  ou volumes que terão propriedades especificas com superficie_n= cell_tags.find(n)
    
    -Emod é uma função que pertence ao espaço de função Q. Isso cria uma função que será usada para representar alguma grandeza física no domínio.
    
    -eletrodo_sup = cell_tags.find(n): Isso parece estar procurando as células com uma determinada tag (marca) igual a 26 no domínio. Essa marcação provavelmente se refere a uma região específica no domínio, que pode ser um eletrodo superior.

    - Emod.x.array[eletrodo_sup] = np.full_like(eletrodo_sup, 1, dtype=PETSc.ScalarType):  Esta linha define os valores da função Emod nas células identificadas como o "eletrodo_sup" (células com a tag 26). Ele define esses valores como 1.0. Isso pode representar algum tipo de condição ou propriedade atribuída à região do eletrodo superior.
"""
Q = fem.FunctionSpace(domain, ("DG", 0))

Emod = fem.Function(Q)
eletrodo_sup = cell_tags.find(26)
Emod.x.array[eletrodo_sup] = np.full_like(eletrodo_sup, 1, dtype=PETSc.ScalarType)
eletrodo_inf = cell_tags.find(27)
Emod.x.array[eletrodo_inf]  = np.full_like(eletrodo_inf, 1, dtype=PETSc.ScalarType)
gel_p=cell_tags.find(28)
Emod.x.array[gel_p]  = np.full_like(gel_p, 0.1, dtype=PETSc.ScalarType)

"""
A determinação das condições de contorno de Dirichletch, podem ser realizadas obtendo-se as faces que serão aplicadas as condições:
    facet_tags.find(n)
"""

dofs_1 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(14))
dofs_2 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(15))
dofs_3 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(16))
dofs_4 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(17))

bcs_1 = fem.dirichletbc(ScalarType(1), dofs_1, V)
bcs_2 = fem.dirichletbc(ScalarType(1), dofs_2, V)
bcs_3 = fem.dirichletbc(ScalarType(1), dofs_3, V)
bcs_4 = fem.dirichletbc(ScalarType(1), dofs_4, V)
bcs= [bcs_1,bcs_2,bcs_3,bcs_4]


"""Definindo os restante das condições para simulação de um problema...."""
V = fem.FunctionSpace(domain, ("CG", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(Emod*ufl.grad(u), ufl.grad(v)) * ufl.dx
x = ufl.SpatialCoordinate(domain)
L = fem.Constant(domain, ScalarType(1)) * v * ufl.dx

problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

"""Vizualização dos resultados"""

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "resultados/subdomain_create.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tags)
    xdmf.write_meshtags(cell_tags)
    xdmf.write_function(uh)

from dolfinx.plot import create_vtk_mesh
# Filter out ghosted cells
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
marker = np.zeros(num_cells_local, dtype=np.int32)
eletrodo_sup = eletrodo_sup[eletrodo_sup<num_cells_local]
eletrodo_inf = eletrodo_inf[eletrodo_inf<num_cells_local]
marker[eletrodo_sup] = 1
marker[eletrodo_inf] = 2
topology, cell_types, x = create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))

p = pyvista.Plotter(window_size=[800, 800])
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Marker"] = marker
grid.set_active_scalars("Marker")
p.add_mesh(grid, show_edges=True)
if pyvista.OFF_SCREEN:
    figure = p.screenshot("subdomains_structured.png")
p.show()

p2 = pyvista.Plotter(window_size=[800, 800])
grid_uh = pyvista.UnstructuredGrid(*create_vtk_mesh(V))
grid_uh.point_data["u"] = uh.x.array.real
grid_uh.set_active_scalars("u")
p2.add_mesh(grid_uh, show_edges=True)
if not pyvista.OFF_SCREEN:
    p2.show()
else:
    figure = p2.screenshot("subdomains_structured2.png")