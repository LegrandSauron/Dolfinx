import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import fem



domain, cell_tags, facet_tags = gmshio.read_from_msh("malhas/malha_com_eletrodo_05.msh", MPI.COMM_WORLD,0, gdim=2)


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


"""Exemplo de implementação"""
Q = fem.FunctionSpace(domain, ("DG", 0))

eletrodo_sup = tag(26)
eletrodo_inf = tag(27)
gel_p= tag(28)

#A ordem de entrada das propriedades na lista deve ser equivalente ao espaço no qual o material ocupa dentro do dominio
material_  = [gel_p,eletrodo_inf,eletrodo_sup]   
Gshear = mat_features(Q,material_, [0.003e-6, 0.034e-6 ,0.2e-6])
Kbulk = mat_features(Q,material_, [1,0.1,0.1])




#A classe Expression foi removida, pois causava dor de cabeça aos desenvolvedores e não está claro o que ela faz. A nova maneira de fazer isso é interpolar uma expressão em uma função: https://fenicsproject.discourse.group/t/equivalent-for-expression-in-dolfinx/2641

from dolfinx.fem import Function,FunctionSpace
from dolfinx.mesh import create_unit_square
from dolfinx.fem import assemble_vector
from ufl import TestFunction, dx, inner
import numpy as np

class MyExpression:
    def __init__(self):
        self.t = 0.0

    def eval(self, x):
        # Added some spatial variation here. Expression is sin(t)*x
        return np.full(x.shape[1], np.sin(self.t)*x[0])

# If your Expression is not spatially dependent, use Constant
# f = Constant(mesh, 0)
# L = inner(f, v)*dx
# f.value = np.sin(2)

V = FunctionSpace(domain, ("CG", 1))
f = MyExpression()
f.t = 0
w = Function(V)
v = TestFunction(V)
L = inner(w, v)*dx
w.interpolate(f.eval)

print(assemble_vector(L).array)
f.t = 2
w.interpolate(f.eval)
print(assemble_vector(L).array)


