import ufl
from petsc4py.PETSc import ScalarType

from dolfinx import fem, mesh, plot
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc 
from petsc4py.PETSc import*
from dolfinx.io import gmshio

dominio, cell_tags, facet_tags = gmshio.read_from_msh("malhas/malha_com_eletrodo_pronta.msh", MPI.COMM_WORLD,0, gdim=2)


"""
Determinando materiais diferentes em uma geometria que possui entidades fisicas pre-definidas:
    -Cria-se um espaço de funções descontinuas para interpolação

    -Determina-se as faces, superficies  ou volumes que terão propriedades especificas com superficie_n= cell_tags.find(n)
    
    -Emod é uma função que pertence ao espaço de função Q. Isso cria uma função que será usada para representar alguma grandeza física no domínio.
    
    -eletrodo_sup = cell_tags.find(n): Isso parece estar procurando as células com uma determinada tag (marca) igual a 26 no domínio. Essa marcação provavelmente se refere a uma região específica no domínio, que pode ser um eletrodo superior.

    - Emod.x.array[eletrodo_sup] = np.full_like(eletrodo_sup, 1, dtype=PETSc.ScalarType):  Esta linha define os valores da função Emod nas células identificadas como o "eletrodo_sup" (células com a tag 26). Ele define esses valores como 1.0. Isso pode representar algum tipo de condição ou propriedade atribuída à região do eletrodo superior.
"""

"""Extrair as tags da malha"""
def tag(n_tag):  
    return cell_tags.find(n_tag)

"Determinação das propriedades de um material "
def mat_features(function_descontinuo,material, constanste):
        space = fem.Function(function_descontinuo)
        for i in range(len(material)):
         
         space.x.array[material[i]]  = np.full_like(material[i],constanste[i], dtype=ScalarType)
        return space
    
Q = fem.FunctionSpace(dominio, ("DG", 0))
eletrodo_sup = tag(26)
eletrodo_inf = tag(27)
gel_p= tag(28)
#A ordem de entrada das propriedades na lista deve ser equivalente ao espaço no qual o material ocupa dentro do dominio
material_  = [gel_p,eletrodo_inf,eletrodo_sup]   
k = mat_features(Q,material_, [0.0005, 0.0000017 ,0.0000017])


x = ufl.SpatialCoordinate(dominio)
ds=ufl.Measure("ds",domain=dominio,subdomain_data=facet_tags) #definindo os subdominios de integração.

# Scaled variable
Temp_init = 300.0
Temp_last =25.0

#time parameters
t_init= 0.0
steps=100
Tempo_final= 1
dt= fem.Constant(dominio,Tempo_final/steps)

#definindo o espaço de funções 
V=fem.FunctionSpace(dominio, ("CG", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)


#Solucao inicial:
u_init = fem.Function(V)
def initial_condition(x, a=0.0):
    return np.exp(-a*(x[0])+ a*x[1])
u_init.interpolate(initial_condition)


"""Definindo as condições de contorno"""

"""
1 14 "eletrodo_superior_l"
1 15 "gel_superior_l"
1 16 "gel_inferior_l"
1 17 "eletrodo_inferior_l"

1 18 "eletrodo_inferior_r"
1 19 "gel_inferior_r"
1 20 "gel_superior_r"
1 21 "eletrodo_superior_r"

1 29 "face_eletrodo_superior"
1 30 "face_eletrodo_superior_baixo"
1 31 "face_eletrodo_baixo_superior"
1 32 "face_eletrodo_baixo_baixo"
2 26 "eletrodo_superior"
2 27 "eletrodo_inferior"
2 28 "gel"
"""

dofs_1 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(14))
dofs_2 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(15))
dofs_3 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(16))
dofs_4 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(17))

bcs_1 = fem.dirichletbc(ScalarType(Temp_init), dofs_1, V)
bcs_2 = fem.dirichletbc(ScalarType(Temp_init), dofs_2, V)
bcs_3 = fem.dirichletbc(ScalarType(Temp_init), dofs_3, V)
bcs_4 = fem.dirichletbc(ScalarType(Temp_init), dofs_4, V)
bc= [bcs_1,bcs_2,bcs_3,bcs_4]


#Salvando os dados 
from dolfinx.io import XDMFFile
with XDMFFile(dominio.comm, "resultados/tcm_2D.xdmf", "w") as xdmf:
    xdmf.write_mesh(dominio)
        
#Defining the body force term
q = fem.Constant(dominio, ScalarType(0.0))

#Forma variacional
a     = ufl.inner(v,u)*ufl.dx + dt*ufl.inner(ufl.grad(v),ufl.grad(u))*k*ufl.dx
L     = dt*ufl.inner(v,q)*ufl.dx + ufl.inner(v,u_init)*ufl.dx

# Solve the linear system
problem = fem.petsc.LinearProblem(a, L, bcs = bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})

for i in range(steps):
    t_init += dt
    uh 	  = problem.solve()
    u_init.interpolate(uh)
    xdmf.write_function(uh,t_init)



