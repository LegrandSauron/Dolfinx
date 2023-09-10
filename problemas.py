
from dolfinx import fem, mesh,log,nls
from dolfinx import*
import numpy as np
import math
# Plotting packages
import matplotlib.pyplot as plt
# Current time package
from datetime import datetime
from sympy import dirichlet_eta
import ufl
from mpi4py import MPI


"""
Construindo a solução para o problema UserExpression 
    Class mat(USerExpression)
"""
mesh = mesh.create_unit_interval(MPI.COMM_WORLD, 32)
V = fem.FunctionSpace(mesh, ("CG", 1))
u = fem.Function(V)


nmesh = 2.0
xL = 1.0
x0 = 0.5
deltax = 1.0
fkls = np.linspace(0, 100)
A = 1.0

class MyExpression0(UserExpression):
    def __init__(self, b, nmesh, **kwargs):
        super().__init__(**kwargs)
        self.A = b
        self.nmesh = nmesh

    def eval(self, value, x):
        lh = (xL - x0) / 2.
        i1 = np.min([np.floor(x[0] / deltax).astype(int), self.nmesh - 1])
        value[0] = .5 * 10 * (x[0] - lh) ** 2 + 200 + self.A * fkls[i1]

    def value_shape(self):
        return ()
 
T_new= fem.Function(V)
T_new.interpolate(MyExpression0)

L = 0.0032 * v.dx(0) * T_new * ufl.dx

"""a"""
def my_expression0(x):
    lh = (xL - x0) / 2.
    i1 = np.minimum(np.floor(x[0] / deltax).astype(int), nmesh - 1).astype(int)
    return .5 * 10 * (x[0] - lh) ** 2 + 200 + A * fkls[i1]

u.interpolate(my_expression0)

x = u.function_space.tabulate_dof_coordinates()
x_order = np.argsort(x[:,0])



class mat(UserExpression): 
    def __init__(self, materials, mat_0, mat_1, mat_2, **kwargs):
        super().__init__(**kwargs)
        self.materials = materials
        self.k_0 = mat_0
        self.k_1 = mat_1
        self.k_2 = mat_2
        
    def eval_cell(self, values, x, cell):
        if self.materials[cell.index] == 0:
            values[0] = self.k_0
        elif self.materials[cell.index] == 1:
            values[0] = self.k_1
        elif self.materials[cell.index] == 4:
            values[0] = self.k_2
        elif self.materials[cell.index] == 5:
            values[0] = self.k_2
        else:
            values[0] = self.k_0
            
    def value_shape(self):
        return () 
    

def my_expression0(x):
    lh = (xL - x0) / 2.
    i1 = np.minimum(np.floor(x[0] / deltax).astype(int), nmesh - 1).astype(int)
    return .5 * 10 * (x[0] - lh) ** 2 + 200 + A * fkls[i1]

u.interpolate(my_expression0)

x = u.function_space.tabulate_dof_coordinates()
x_order = np.argsort(x[:,0])

