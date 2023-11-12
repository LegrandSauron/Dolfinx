"""https://fenicsproject.discourse.group/t/implement-assign-in-dolfinx/12022"""


"""Olá,

Ao tentar converter um código dolfin antigo em dolfinx, estou me perguntando como substituir o assigncomando que usei para inicializar os valores de uma função com o resultado de um cálculo anterior, por exemplo:"""
import dolfinx

# Fenics-related packages
from dolfinx import fem, nls, la
import numpy as np
import math

# Plotting packages
import matplotlib.pyplot as plt

# Current time package
from datetime import datetime
from mpi4py import MPI


L = 1
d = L/10.
h = d/6. 

my_domain = dolfinx.mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, -0.5*d), (L, 0.5*d)), n=(int(L/h), int(d/h)),
                            cell_type=dolfinx.mesh.CellType.triangle)

V = dolfinx.fem.VectorFunctionSpace(my_domain, ("CG", 2))
u = dolfinx.fem.Function(V)
u_saved = dolfinx.fem.Function(V)

### follows a calculation: u is the solution

#u_saved.assign(u) #old dolfin command which I try to replace


u_saved.x.array[:] = u.x.array #SOLUCAO 