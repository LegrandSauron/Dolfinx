from scripts import*
import ufl
from dolfinx import fem
from petsc4py.PETSc import*

class Hencky_Strain:
    def __init__(self,Func_space):
        self.V = Func_space

    def u(self):
        return fem.Function(self.V)
        
    def identidade(self):
        return ufl.Identity((len(self.u())))

    def extract_space(self):
        return self.V.ufl_function_space
    
    def tensor(self):
        F= (self.identidade() + ufl.grad(self.u()))
        C= (F*F.T)
        N, M = C.ufl_shape
        H= ufl.as_tensor([[(1/2)*(ufl.ln(C[i,j])) for i in range(N)] for j in range(M)])
        return H
