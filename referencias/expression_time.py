from dolfinx.io import gmshio
from petsc4py.PETSc import ScalarType
import ufl
from dolfinx import fem
import numpy as np
import math
# Plotting packages
import matplotlib.pyplot as plt
# Current time package
from datetime import datetime
from mpi4py import MPI

domain, cell_tags, facet_tags = gmshio.read_from_msh("malhas/malha_com_eletrodo_05.msh", MPI.COMM_WORLD,0, gdim=2)

"""Ok, finalmente encontrei uma solução para uma dependência de tempo em uma forma variacional que não leva a sobrecargas do compilador que gostaria de compartilhar.

Defina uma classe de expressão para seu valor dependente do tempo:"""

class load_expr:
    def __init__(self):
        self.t = 0.0

    def eval(self, x):
        pr = 10.0
        # Added some spatial/temporal variation here
        return np.full(x.shape[1], pr*self.t/T)
    
#Defina um espaço funcional real (um grau de liberdade global):

V_real = FunctionSpace(mesh, ("R", 0))

#Interpole a expressão no espaço de funções real:

load = load_expr()
load.t = 0.0
load_func = Function(V_real)
load_func.interpolate(load.eval)
#Na sua forma variacional (pode ser definida fora do loop temporal):
varform = ... + load_func*J*dot(dot(inv(F).T,n0), var_u)*ds4

#dentro do loop temporal, ligue
while ...:
    load.t = t
    load_func.interpolate(load.eval)
#para atualizar a expressão.

"""Assim, não há compilação excessiva de formulários repetidos após cada etapa de tempo, o que acontecia se eu tivesse um Constant(mesh, 10.*t/T)retorno para cada etapa.
(Talvez isso pudesse ter sido contornado com a atualização da Constante com .value, no entanto, isso produziu o segfault conforme mencionado acima.)"""



"""Olá, acabei de encontrar este tópico porque estou tentando impor uma condição de contorno de Dirichlet não dependente do espaço que varia com o tempo. Aqui está um exemplo do meu código: https://fenicsproject.discourse.group/t/equivalent-for-expression-in-dolfinx/2641/22"""


class TimeDepBC():
    def __init__(self, t=0):
        self.t = t
        self.values = None
    def __call__(self, x):
        if self.values is None:
            self.values = np.zeros((2, x.shape[1]), dtype=ScalarType)
        # Update values[0] and values[1] (the x and y component depending on the self.t parameter)
        # ...
        # For instance:
        self.values[0] = self.t
        return self.values

V = ufl.FiniteElement(domain,("CG", 1))
bc_expr = TimeDepBC()
u_right = fem.Function(V)
u_right.interpolate(bc_expr.__call__)
print(u_right.x.array)
bc_expr.t = 5
u_right.interpolate(bc_expr.__call__)
print(u_right.x.array)

#Como representar esta merda ?
disp =fem.Expression("0.5*t/Tramp")

disp2 =fem.Expression("-0.5*t/Tramp")

phiRamp =fem.Expression("(250/phi_norm)*t/Tramp")


