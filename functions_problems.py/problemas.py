from dolfinx import fem, nls, la,mesh
import numpy as np
import math

# Plotting packages
# Current time package
from mpi4py import MPI
import ufl


# Dimensions
scaleX = 1.0e4
xElem = 1

scaleY = 1400.e0 

scaleZ = 1.0e4
zElem = 1

# N number of elements in y-direction                                             
N = 128

#scaleZ = 30.e0
int1 = 200.e0
int2 = scaleY/2.-int1
int3 = int1/2.0 + int2
int4 = 500.e0
                
M1 = 60. #N/scaleY*int1
M2 = 2.0 #90 #int2/scaleY*N
M3 = M1/2
r1 = 1/1.5
r2 = 1/1.06     
r3 = r1
a1 = (1-r1)/(1-r1**M1)
a2 = (1-r2)/(1-r2**(M2)) 
a3 = (1-r3)/(1-r3**M3)
                
preMapLength = float(int1 + 2*M2*(int1/M1))

domain =mesh.create_box(MPI.COMM_WORLD,[(0.,-preMapLength, 0.0),(scaleX, preMapLength, scaleZ)],[xElem,N, zElem])


x = ufl.SpatialCoordinate(domain)


tol=1e-5
tol = 1.e-5


import math

class CustomExpression():
    def __init__(self, int2, scaleY, tol, cPos0, cNum):
        self.int2 = int2
        self.scaleY = scaleY
        self.tol = tol
        self.cPos0 = cPos0
        self.cNum = cNum
        
    def eval(self, value, x):
        if abs(x[1]) >= self.int2 - self.tol and abs(x[1]) <= self.scaleY/2 + self.tol:
            value[0] = math.log(self.cPos0)
        else:
            value[0] = math.log(self.cNum)

# Uso da classe CustomExpression
int2 = 1.0  # Substitua pelo valor adequado
scaleY = 2.0  # Substitua pelo valor adequado
tol = 0.01  # Substitua pelo valor adequado
cPos0 = 10.0  # Substitua pelo valor adequado
cNum = fem.DOLFIN_EPS  # Substitua pelo valor adequado
init_omgPos = CustomExpression(int2, scaleY, tol, cPos0, cNum)



