"""
Code for hydrogel ionotronics.

- with the model comprising:
    > Large deformation compressible Gent elasticity
    > Dilute-solution mixing model for the 
        diffusing cation and anion
        and
    > Electro-quasistatics
    

- with the numerical degrees of freedom:
    > vector displacements
    > scalar electrochemical potentials
    > scalar electrostatic potential.
    
- with basic units:
    > Length: mu-m
    >   Time: mu-s
    >   Mass: mg
    >  Moles: n-mol
    > Charge: mC
    
    Eric M. Stewart    and    Sooraj Narayan,   
   (ericstew@mit.edu)        (soorajn@mit.edu)     
    
                   Fall 2022 
                   
   
Note: This code will yield revelant results from the ionotronics paper, but is
  not yet fully cleaned up. In the near future I plan to make edits to render 
  it more readable, remove extraneous features, and provide more detailed 
  explanatory comments.

                   
Code acknowledgments:
    
    - Jeremy Bleyer, whose `elastodynamics' code was a useful reference for 
      constructing our own implicit dynamic solution procedure.
     (https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html)
         
    
"""


# Fenics-related packages
from dolfinx import fem, nls, la
import numpy as np
import math

# Plotting packages
import matplotlib.pyplot as plt

# Current time package
from datetime import datetime
from mpi4py import MPI

# Set level of detail for log messages (integer)
#
# Guide:
# CRITICAL  = 50, // errors that may lead to data corruption
# ERROR     = 40, // things that HAVE gone wrong
# WARNING   = 30, // things that MAY go wrong later
# INFO      = 20, // information of general interest (includes solver info)
# PROGRESS  = 16, // what's happening (broadly)
# TRACE     = 13, // what's happening (in detail)
# DBG       = 10  // sundry


# -----------------------------------------------------------
# Global Fenics parameters
# parameters["form_compiler"]["cpp_optimize"]=True
# parameters["form_compiler"]["optimize"] = True
# set_log_level(30)

# The behavior of the form compiler FFC can be adjusted by prescribing
# various parameters. Here, we want to use the UFLACS backend of FFC::

# Optimization options for the form compiler


# Dimensions

from dolfinx.io import gmshio
from petsc4py.PETSc import ScalarType
import ufl

domain, cell_tags, facet_tags = gmshio.read_from_msh(
    "malhas/malha_com_eletrodo_pronta.msh", MPI.COMM_WORLD, 0, gdim=2
)


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

def mat_features(function_descontinuo, material, constanste):
    space = fem.Function(function_descontinuo)
    for i in range(len(material)):
        space.x.array[material[i]] = np.full_like(material[i], constanste[i], dtype=ScalarType)
    return space


"""Exemplo de implementação"""
Q = fem.FunctionSpace(domain, ("DG", 0))

eletrodo_sup = tag(26)
eletrodo_inf = tag(27)
gel_p = tag(28)

# A ordem de entrada das propriedades na lista deve ser equivalente ao espaço no qual o material ocupa dentro do dominio
material_ = [gel_p, eletrodo_inf, eletrodo_sup]
Gshear = mat_features(Q, material_, [0.003e-6, 0.034e-6, 0.2e-6])
Kbulk = mat_features(Q, material_, [2000 * 0.003e-6, 2000 * 0.034e-6, 2000.0 * 0.2e-6])
intInd = mat_features(Q, material_, [1, 0, 0])
matInd = mat_features(Q, material_, [1, 0, 0])
Gshear0 = 100.0e-6  # uso na formução fraca e no cconstrutor
Im_gent = mat_features(Q, material_, [300, 90.0, 90.0])


D = 1.0e-2  # 1.0e0                 # Diffusivity [L2T-1]
RT = 8.3145e-3 * (273.0 + 20.0)  # Gas constant*Temp [ML2T-2#-1]
Farad = 96485.0e-6  # Faraday constant [Q#-1]

""" Initial concentrations"""
cPos0 = 0.274  # Initial concentration [#L-3]
cNeg0 = cPos0  # Initial concentration [#L-3]
cMax = 10000 * 1e-9  # 0.00001*1e-9 # 10000e0*1.e-9

"""Eletrical permittivity """
vareps0 = fem.Constant(domain, 8.85e-12 * 1e-6)  #
vareps_num = mat_features(Q, material_, [1.0e4, 1.0, 1.0])  # Permissividade do gel igual a 10000
vareps_r = mat_features(Q, material_, [80, 6.5, 6.5])  # eletrical permittivity for material
vareps = vareps0 * vareps_r * vareps_num

# Mass density
rho = 1e-9  # 1e3 kg/m^3 = 1e-9 mg/um^3,

# Rayleigh damping coefficients
eta_m = 0.00000  # Constant(0.000005)
eta_k = 0.00000  # Constant(0.000005)


""" Generalized-alpha method parameters """
alpha = 0.0
gamma = 0.5 + alpha
beta = ((gamma + 0.5) ** 2) / 4


""" Simulation time related params (reminder: in microseconds)"""
ttd = 0.01
# Step in time
t = 0.0  # initialization of time
T_tot = 0e6 * 1.0e6  # 200.0 #0.11        # total simulation time

#

t1 = 15.0e6  # sem uso
t2 = 35.0e6  # sem uso
t3 = 2.5e6  # sem uso
t4 = 52.5e6  # sem uso


T2_tot = 30.0e6  # 0.0001*1.e6 #t1+t2+t3+t4
dt = T2_tot / 500  # incrementos de tempo

phi_norm = RT / Farad  # "Thermal" Volt

# Define function space, scalar
U2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)  # Displacent
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)  # concentrações + - , electric potential
D0 = ufl.FiniteElement("DG", domain.ufl_cell(), 0)
D1 = ufl.FiniteElement("DG", domain.ufl_cell(), 1)

# DOFs
TH = ufl.MixedElement([U2, P1, P1, P1])

ME = fem.FunctionSpace(domain, TH)  # Total space for all DOFs
#Extraindo os sub espaços do elemento misto e os mapas contendo os graus de liberdade 
num_subs = ME.num_sub_spaces
spaces = []
maps = []
for i in range(num_subs):
    space_i, map_i = ME.sub(i).collapse()
    spaces.append(space_i)
    maps.append(map_i)




W = fem.FunctionSpace(domain, P1)  # Scalar space for visualization later
W2 = fem.FunctionSpace(domain, U2)  # Vector space for visualization later
W3 = fem.FunctionSpace(domain, D0)  # DG space for visualization later
W4 = fem.FunctionSpace(domain, D1)  # DG space for visualization later

# Define test functions in weak form
dw = ufl.TrialFunction(ME)

(u_test, omgPos_test, phi_test, omgNeg_test) = ufl.TestFunctions(ME)  # Test function

# Define actual functions with the required DOFs
w = fem.Function(ME)
(u, omgPos, phi, omgNeg) = ufl.split(w)  # current DOFs

# A copy of functions to store values in last step for time-stepping.
w_old = fem.Function(ME)
(u_old, omgPos_old, phi_old, omgNeg_old) = ufl.split(w_old)  # old DOFs

v_old = fem.Function(W2)
a_old = fem.Function(W2)

# Initial chemical potential
mu0 = ufl.ln(cPos0)
mu20 = ufl.ln(cNeg0)



