
# Fenics-related packages
import ufl
from dolfinx import fem
import numpy as np
import math
# Plotting packages
import matplotlib.pyplot as plt
# Current time package

from dolfinx.io import gmshio
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import pyvista

domain, cell_tags, facet_tags = gmshio.read_from_msh("malhas/malha_com_eletrodo_05.msh", MPI.COMM_WORLD,0, gdim=2)
V= fem.FunctionSpace(domain, ("CG", 1))
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


Emod = fem.Function(Q) #U2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1) #<vector element with 2 components of <CG1 on a quadrilateral>>
Kbulk_space= fem.Function(Q)

"""Definindo as funções para determinar espaço de cada "material no dominio"""
def tag(n_tag):
    return cell_tags.find(n_tag)

def mat_features(function,material, constanste):
        for i in range(len(material)):
         function.x.array[material[i]]  = np.full_like(material[i],constanste[i], dtype=ScalarType)




eletrodo_sup = tag(26)
eletrodo_inf = tag(27)
gel_p= tag(28)

#A ordem de entrada das propriedades na lista deve ser equivalente ao espaço no qual o material ocupa dentro do dominio
material_  = [gel_p,eletrodo_inf,eletrodo_sup]   

Gshear = mat_features(Emod,material_, [0.003e-6, 0.034e-6 ,0.2e-6])
Kbulk = mat_features(Kbulk_space,material_, [2000*0.003e-6,2000*0.034e-6,2000.0*0.2e-6])


"""
A determinação das condições de contorno de Dirichletch, podem ser realizadas obtendo-se as faces que serão aplicadas as condições:
    facet_tags.find(n)
"""

dofs_1 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(14))
dofs_2 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(15))
dofs_3 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(16))
dofs_4 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(17))

bcs_1 = fem.dirichletbc(ScalarType(0), dofs_1, V)
bcs_2 = fem.dirichletbc(ScalarType(0), dofs_2, V)
bcs_3 = fem.dirichletbc(ScalarType(0), dofs_3, V)
bcs_4 = fem.dirichletbc(ScalarType(0), dofs_4, V)
bcs= [bcs_1,bcs_2,bcs_3,bcs_4]


"""Definindo os restante das condições para simulação de um problema...."""

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx 
x = ufl.SpatialCoordinate(domain)
f=   2*(ufl.pi**2)*ufl.sin(ufl.pi*x[0])*ufl.sin(2*ufl.pi*x[1])         #2π2sin(πx)sin(2πy) 
L = f* v * ufl.dx
problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

"""Vizualização dos resultados"""

uh.name = "deslocamento"
Emod.name = 'gshear'
Kbulk_space.name = "Kbulk"
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "resultados/subdomain_create.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tags)
    xdmf.write_meshtags(cell_tags)
    xdmf.write_function(uh)
    xdmf.write_function(Emod)
    xdmf.write_function(Kbulk_space)

from dolfinx.plot import create_vtk_mesh
# Filter out ghosted cells
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
marker = np.zeros(num_cells_local, dtype=np.int32)
eletrodo_sup = eletrodo_sup[eletrodo_sup<num_cells_local]
eletrodo_inf = eletrodo_inf[eletrodo_inf<num_cells_local]
gel_p= gel_p[gel_p<num_cells_local]
marker[eletrodo_sup] = 1
marker[eletrodo_inf] = 2
marker[gel_p] = 3
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
    pass
    p2.show()
else:
    figure = p2.screenshot("subdomains_structured2.png")

