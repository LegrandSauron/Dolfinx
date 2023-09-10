"""This code is impementing the mecanical part of capacitive strain sensor """

from dolfinx import fem, mesh,log,nls
import numpy as np
import math
# Plotting packages
import matplotlib.pyplot as plt
# Current time package
from datetime import datetime
from sympy import dirichlet_eta
import ufl

"""
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

"""
   
materials = MeshFunction("size_t", domain, domain.topology().dim(), 0)

# Material parameters
Gshear = mat(materials, fem.Constant(0.003e-6), fem.Constant(0.034e-6), fem.Constant(0.2e-6), degree=0)  
Kbulk = mat(materials, fem.Constant(2000*0.003e-6), fem.Constant(2000*0.034e-6),fem.Constant(2000.0*0.2e-6), degree=0)  
Gshear0 = 100.0e-6 
Im_gent = mat(materials, fem.Constant(300), fem.Constant(90.0), fem.Constant(90.0), degree=0)
  
D = 1.e-2 #1.0e0                 # Diffusivity [L2T-1]
RT = 8.3145e-3*(273.0+20.0)      # Gas constant*Temp [ML2T-2#-1]
Farad = 96485.e-6                # Faraday constant [Q#-1]
#L_debye = 0.01*scaleY # 6e-3 #12.0e-3 #600.0e-3
# Initial concentrations
cPos0 = 0.274                # Initial concentration [#L-3]
cNeg0 = cPos0                # Initial concentration [#L-3]
cMax = 10000*1e-9 #0.00001*1e-9 # 10000e0*1.e-9

vareps0 = fem.Constant(8.85e-12*1e-6)
vareps_num =  mat(materials, fem.Constant(1.0e4), fem.Constant(1.0), fem.Constant(1.0), degree=1)
vareps_r = mat(materials, fem.Constant(80), fem.Constant(6.5), fem.Constant(6.5))
vareps = vareps0*vareps_r*vareps_num
#vareps = Constant(Farad*Farad*(cMax*(cPos0+cNeg0))/RT*L_debye*L_debye)

# Mass density
rho = fem.Constant(1e-9)  # 1e3 kg/m^3 = 1e-9 mg/um^3,

# Rayleigh damping coefficients
eta_m = fem.Constant(0.00000) # Constant(0.000005)
eta_k = fem.Constant(0.00000) # Constant(0.000005)



# Quick-calculate sub-routines

phi_norm = RT/Farad # "Thermal" Volt



def F_calc(u):
    dim = len(u)
    Id = ufl.Identity(dim) # Identity tensor
    
    F = Id + ufl.grad(u) # 3D Deformation gradient
    return F # Full 3D F

    
def Piola(F,phi):
    
    Id = ufl.Identity(3)
    J = ufl.det(F)
    
    C = F.T*F
    Cdis = J**(-2/3)*C
    I1 = ufl.tr(Cdis)
         
    eR = -phi_norm*ufl.grad(phi)
    e_sp = ufl.inv(F.T)*eR
    #
    T_max = vareps0*vareps_r*J*(ufl.outer(e_sp,e_sp) - 1/2*(ufl.inner(e_sp,e_sp))*Id)*inv(F.T) 
    
    # Piola stress (Gent)
    TR = J**(-2/3)*Gshear*(Im_gent/(Im_gent+3-I1))*(F - 1/3*ufl.tr(Cdis)*ufl.inv(F.T)) + Kbulk*ufl.ln(J)*ufl.inv(F.T) + T_max
    
    return TR
