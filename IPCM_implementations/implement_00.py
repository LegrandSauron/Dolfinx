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
        space.x.array[material[i]] = np.full_like(
            material[i], constanste[i], dtype=ScalarType
        )
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
vareps_num = mat_features(
    Q, material_, [1.0e4, 1.0, 1.0]
)  # Permissividade do gel igual a 10000
vareps_r = mat_features(
    Q, material_, [80, 6.5, 6.5]
)  # eletrical permittivity for material
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


"""Ate aqui, tudo ok """
"""Não sei a finalidade, mas a interpretação do argumento de fem.Expression() é dado por : https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html"""

init_omgPos = fem.Expression('abs(x[1])>=int2-tol && abs(x[1])<=scaleY/2+tol?std::log((cPos0)):std::log((cNum))', int2=int2 scaleY=scaleY, tol = tol, cPos0 = cPos0, cNum=DOLFIN_EPS, degree=0)

omgPos_init = interpolate(init_omgPos,ME.sub(1).collapse())
#assign(w_old.sub(1),omgPos_init)
w_old.sub(1).x.array[:]= omgPos_init.x.array 


init_omgNeg = Expression('abs(x[1])>=int2-tol && abs(x[1])<=scaleY/2+tol?std::log((cNeg0)):std::log((cNum))', int2=int2, scaleY=scaleY, tol = tol, cNeg0 = cNeg0,  cNum=DOLFIN_EPS, degree=0)
omgNeg_init = interpolate(init_omgNeg,ME.sub(3).collapse())
assign(w_old.sub(3),omgNeg_init)

# Update initial guess for w
assign(w.sub(3),omgNeg_init)
assign(w.sub(1),omgPos_init)
#assign(w.sub(5),cNeg_init)
#assign(w.sub(2),cPos_init)




cPos = ufl.exp(omgPos - Farad * phi * phi_norm / RT)
cNeg = ufl.exp(omgNeg + Farad * phi * phi_norm / RT)
cPos_old = ufl.exp(omgPos_old - Farad * phi_old * phi_norm / RT)
cNeg_old = ufl.exp(omgNeg_old + Farad * phi_old * phi_norm / RT)

# Quick-calculate sub-routines


def F_calc(u):
    dim = len(u)
    Id = ufl.Identity(dim)  # Identity tensor

    F = Id + ufl.grad(u)  # 3D Deformation gradient
    return F  # Full 3D F


def Piola(F, phi):
    Id = ufl.Identity(2)
    J = ufl.det(F)

    C = F.T * F
    Cdis = J ** (-2 / 3) * C
    I1 = ufl.tr(Cdis)

    eR = -phi_norm * ufl.grad(phi)
    e_sp = ufl.inv(F.T) * eR
    #
    T_max = (
        vareps0
        * vareps_r
        * J
        * (ufl.outer(e_sp, e_sp) - 1 / 2 * (ufl.inner(e_sp, e_sp)) * Id)
        * ufl.inv(F.T)
    )

    # Piola stress (Gent)
    TR = (
        J ** (-2 / 3)
        * Gshear
        * (Im_gent / (Im_gent + 3 - I1))
        * (F - 1 / 3 * ufl.tr(Cdis) * ufl.inv(F.T))
        + Kbulk * ufl.ln(J) * ufl.inv(F.T)
        + T_max
    )

    return TR


# variable time step
dk = fem.Constant(domain, 0.0)


# Update formula for acceleration
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_aceleration(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dk
        beta_ = beta
    else:
        dt_ = float(dk)
        beta_ = float(beta)
    return (u - u_old - dt_ * v_old) / beta_ / dt_**2 - (
        1 - 2 * beta_
    ) / 2 / beta_ * a_old


# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_velocity(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dk
        gamma_ = gamma
    else:
        dt_ = float(dk)
        gamma_ = float(gamma)
    return v_old + dt_ * ((1 - gamma_) * a_old + gamma_ * a)


def update_fields(u_proj, u_proj_old, v_old, a_old):
    """Update fields at the end of each time step."""

    # Get vectors (references)
    u_vec, u0_vec = u_proj.vector(), u_proj_old.vector()
    v0_vec, a0_vec = v_old.vector(), a_old.vector()

    # use update functions using vector arguments
    a_vec = update_aceleration(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_velocity(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

    # Update (u_old <- u)
    v_old.vector()[:], a_old.vector()[:] = v_vec, a_vec
    # u_old.vector()[:] = u_proj.vector()


def ppos(x):
    return (x + abs(x)) / 2.0


def avg(x_old, x_new, alpha):  # atualização dos campos de velocidade e deslocamento
    return alpha * x_old + (1 - alpha) * x_new


# Newmark-beta kinematical update
a_new = update_aceleration(u, u_old, v_old, a_old, ufl=True)
v_new = update_velocity(a_new, u_old, v_old, a_old, ufl=True)

# get avg fields for generalized-alpha method
u_avg = avg(u_old, u, alpha)
v_avg = avg(v_old, v_new, alpha)
omgPos_avg = avg(omgPos_old, omgPos, alpha)
phi_avg = avg(phi_old, phi, alpha)
omgNeg_avg = avg(omgNeg_old, omgNeg, alpha)

# Explicit concentration updates
cPos = ufl.exp(omgPos_avg - phi_avg)
cNeg = ufl.exp(omgNeg_avg + phi_avg)
cPos_old = ufl.exp(omgPos_old - phi_old)
cNeg_old = ufl.exp(omgNeg_old + phi_old)

"""Construindo os tensores de deformação F com os valores de u_avg= , u_old= antigo"""
# Kinematics
F = F_calc(u_avg)
C = F.T * F
Ci = ufl.inv(C)
F_old = F_calc(u_old)  # grad(u_old) + Identity(3)
J = ufl.det(F)
J_old = ufl.det(F_old)


"""""" """""" """""" """''
       WEAK FORMS
""" """""" """""" """""" ""

dynSwitch = fem.Constant(domain, 1.0)

L0 = (
    ufl.inner(Piola(F_calc(u_avg), phi_avg), ufl.grad(u_test)) * ufl.dx
    + dynSwitch * rho * ufl.inner(a_new, u_test) * ufl.dx
)

L1 = (
    ufl.dot((cPos - cPos_old) / dk / D, omgPos_test) * ufl.dx
    + (cPos_old)
    * ufl.inner(Ci * matInd * ufl.grad(omgPos_avg), ufl.grad(omgPos_test))
    * ufl.dx
)

L3 = (
    ufl.dot(vareps * phi_norm * J * Ci * ufl.grad(phi_avg), ufl.grad(phi_test)) * ufl.dx
    - ufl.dot(Farad * matInd * cMax * (cPos - cNeg), phi_test) * ufl.dx
)

L4 = (
    ufl.dot((cNeg - cNeg_old) / dk / D, omgNeg_test) * ufl.dx
    + (cNeg_old)
    * ufl.dot(Ci * matInd * ufl.grad(omgNeg_avg), ufl.grad(omgNeg_test))
    * ufl.dx
)


# Total weak form

L = (1 / Gshear0) * L0 + L1 + L3 + L4
"""
L = L0 + L1 + L2 + L3
"""
# Automatic differentiation tangent:
a = ufl.derivative(L, w, dw)


# Boundary condition expressions as necessary

disp = fem.Expression(("0.5*t/Tramp"), Tramp=T2_tot - T_tot, t=0.0, degree=1)
#disp2 = fem.Expression(("-0.5*t/Tramp"), Tramp=T2_tot - T_tot, t=0.0, degree=1)

phiRamp = fem.Expression(
    ("(250/phi_norm)*t/Tramp"), phi_norm=phi_norm, Tramp=T_tot, t=0.0, degree=1
)

# Boundary condition definitions

bcs_1 = fem.dirichletbc(ME.sub(2), 0.0, facets, 8)  # Ground center of device
bcs_2 = fem.dirichletbc(ME.sub(2), 0.0, facets, 6)  # Ground top of device
bcs_3 = fem.dirichletbc(ME.sub(2), 0.0, facets, 3)  # Ground bottom of device
bcs_4 = fem.dirichletbc(ME.sub(2), phiRamp, facets, 6)  # Ground bottom of device
bcs_a = fem.dirichletbc(ME.sub(0).sub(0), 0.0, facets, 2)
Left().mark(facets, 2)
Bottom().mark(facets, 3)
Top().mark(facets,6)
Center().mark(facets,8)



x_plot = scaleX
phi_eq = w_old(x_plot, scaleY / 2, scaleZ / 2)[4]
# phiRamp = Expression(("phi_eq + 2.0e3/phi_norm*(1-exp(-10*t/Tramp)) + 1.e3/phi_norm*sin(2*pi*f*t/1e6)"),
#                 phi_eq=phi_eq, phi_norm = phi_norm, pi=np.pi, f=100,  Tramp = T2_tot, t = 0.0, degree=1)
phiRamp = Expression(
    ("min(1.0e3/phi_norm*(t/Tramp), 1.0e3/phi_norm)"),
    phi_eq=phi_eq,
    phi_norm=phi_norm,
    pi=np.pi,
    Tramp=T2_tot,
    t=0.0,
    degree=2,
)

disp = Expression(("5.5*scaleX*t/Tramp"), scaleX=scaleX, Tramp=T2_tot, t=0.0, degree=1)
bcs_f = fem.dirichletbc(ME.sub(2), phiRamp, facets, 6)  # Ramp up phi on top face

# bcs_b1 = fem.dirichletbc(ME.sub(0),Constant((0.,0., 0.)),facets,2) # left face built-in
bcs_b1 = fem.dirichletbc(ME.sub(0).sub(0), 0.0, facets, 2)  # pull right face
bcs_b2 = fem.dirichletbc(ME.sub(0).sub(0), disp, facets, 7)  # pull right face
bcs_b3 = fem.dirichletbc(ME.sub(0).sub(1), 0.0, facets, 10)
bcs_b3a = fem.dirichletbc(ME.sub(0).sub(0), 0.0, facets, 10)
bcs_b3b = fem.dirichletbc(ME.sub(0).sub(2), 0.0, facets, 10)
bcs_b3c = fem.dirichletbc(ME.sub(0).sub(0), 0.0, facets, 9)
bcs_b3d = fem.dirichletbc(ME.sub(0).sub(2), 0.0, facets, 9)
bcs_b4 = fem.dirichletbc(ME.sub(0).sub(2), 0.0, facets, 5)

Left().mark(facets, 2)
Right().mark(facets,7)

Back().mark(facets, 5)
TopTop().mark(facets, 9)
BottomBottom().mark(facets, 10)




# bcs2 = [bcs_3, bcs_a, bcs_b1, bcs_b3, bcs_b3a, bcs_b3b, bcs_b3c, bcs_b3d, bcs_b4, bcs_f]

bcs2 = [bcs_3, bcs_a, bcs_b1, bcs_b3, bcs_b4, bcs_f]


""" Output file setup"""
# file_results = XDMFFile("results/suo_capacitor_3D_traction.xdmf")
# file_results.parameters["flush_output"] = True
# file_results.parameters["functions_share_mesh"] = True


# initialize counter
ii = 0
"""""" """""" """""" """
    RUN ANALYSIS
""" """""" """""" """"""
# u_v = w_old.sub(0)
# u_v.rename("disp", "")

# omgPos_v = w_old.sub(1)
# omgPos_v.rename("omega", "")

# phi_v = w_old.sub(2)
# phi_v.rename("phi", "")

# omgNeg_v = w_old.sub(3)
# omgNeg_v.rename("omega2", "")

# Gshear_v = project(1e6 * Gshear, W3)
# Gshear_v.rename("Gshear", "")
# file_results.write(Gshear_v, t)

# Kbulk_v = project(1e6 * Kbulk, W3)
# Kbulk_v.rename("Kbulk", "")
# file_results.write(Kbulk_v, t)

# intInd_v = project(intInd, W3)
# intInd_v.rename("IntIndex", "")
# file_results.write(intInd_v, t)


# Initialize arrays for plotting later
Ny = N + 1
x_plot = scaleX
y = np.sort(
    np.array(mesh.coordinates()[np.where(mesh.coordinates()[:, 0] == x_plot), 1])
)  # np.linspace(-scaleY/2, scaleY/2, Ny)
y = y[0, :]
y2 = np.linspace(0, scaleY / 10, Ny)

output_step = [0.0, 0.001, 0.005, 5.01, 50.01]
op_dim = len(output_step)
phi_var = np.zeros((Ny, op_dim))
cPos_var = np.zeros((Ny, op_dim))
# cNeg_var = np.zeros((Ny, op_dim))

voltage_out = np.zeros(100000)
charge_out = np.zeros(100000)
disp_out = np.zeros(100000)
time_out = np.zeros(100000)
trac_out = np.zeros(100000)

i = 0
ii = 0

trac_out = np.zeros(100000)
ocv_out = np.zeros(100000)
t_out = np.zeros(100000)
t = T_tot
jj = 0
spike = 0

# Turn on implicit dynamics
dynSwitch.assign(Constant(1.0))

voltage_out = np.zeros(100000)
charge_out = np.zeros(100000)
disp_out = np.zeros(100000)
time_out = np.zeros(100000)
trac_out = np.zeros(100000)

while round(t, 2) <= round(T_tot + 2.0 * T2_tot, 2):
    # Output storage
    # if ii%10 == 0:
  #  if True:
        # file_results.write(u_v, t)
        # file_results.write(omgPos_v, t)
        # file_results.write(phi_v, t)
        #phi_nonDim = phi_norm / 1.0e3 * phi
       # phi_nonDim = project(phi_nonDim, W)
        #phi_nonDim.rename("phi_Orig", "")
        # file_results.write(phi_nonDim, t)
        # file_results.write(omgNeg_v, t)
        # file_results.write(Gshear_v, t)

        #
        #_w_0, _w_1, _w_2, _w_3 = w_old.split()
    """    
    if (t >= T_tot+(spike+1)*T2_tot):
        spike+=1
    # variable time-stepping
    #dt = 0.5 
    if ((t-T_tot-spike*T2_tot) < (t1/10)):
        dt = t1/500
    #elif t-T_tot-spike*T2_tot < ((t1+t2)/100):
    #    dt = t1/100
    #elif t-T_tot-spike*T2_tot < ((t1+t2)/100)
    else:
        dt = t1/500 #T2_tot/500
    """

    dt = T2_tot / 20

    # dt = T2_tot/1000
    dk.assign(dt)

    # disp2.t = min(t - T_tot, T2_tot)
    phiRamp.t = t - T_tot - float(alpha * dt)

    trac_final = 0.040e-6  # 40 kPa total
    n = ufl.FacetNormal(domain)


    if t - T_tot <= T2_tot:
        trac_value = 0.0
    else:
        trac_value = trac_final * (t - T_tot - T2_tot - float(alpha * dt)) / T2_tot

    trac = -trac_value * n
    trac_out[jj] = trac_value

    Ltrac = -ufl.dot(trac / Gshear0, u_test) * ds(9)  # + dot(trac/Gshear0, u_test)*ds(3)

    # Set up the non-linear problem (free swell)
    StressProblem = fem.petsc.NonlinearProblem(L + Ltrac, w, bcs2, J=a)
    

    # Set up the non-linear solver
    solver =nls.petsc.NewtonSolver(MPI.COMM_WORLD,StressProblem)
# Set Newton solver options
#solver.convergence_criterion = "residual"
    solver.rtol = 1.0e-6
    #solver.report = True
    solver.atol = 1.0e-6
    solver.max_it = 60

    """COnfigurações para o solver não linear  :
    solver.solver_type = "gmres"
    #solver.preconditioner_type = "lu"
    #solver.initial_guess = None  # Pode ser um vetor ou None
    #solver.divergence_tolerance = 1e-4
    #solver.monitor = None  # Pode ser uma função de monitoramento personalizada
    #solver.line_search = True
    #solver.jacobian_update = "approximate"

    #solver.error_on_nonconvergence = True
    """

    """COnfigurações para o solver linear  :
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}pc_type"] = "gamg"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()
    """

    # Solve the problem
    (iter, converged) = solver.solve()

    # output measured device voltage, etc.
    phi_top = w(x_plot, scaleY / 2, scaleZ / 2)[4] * phi_norm
    phi_btm = w(x_plot, -scaleY / 2, scaleZ / 2)[4] * phi_norm
    voltage_out[ii] = phi_top - phi_btm
    disp_out[ii] = w(scaleX, scaleY / 2 - slope * scaleY, scaleZ / 2)[0]
    time_out[ii] = t - float(alpha * dt)
    # trac_out[jj]    = trac_final*(t-T_tot)/(T2_tot)
    ocv_out[jj] = phi_top - phi_btm
    t_out[jj] = t - T_tot - float(alpha * dt)
    #
    charge_out[ii] = assemble(Farad * cMax * (cPos - cNeg) * dx(3))

    u_proj = project(u, W2)
    u_proj_old = project(u_old, W2)
    update_fields(u_proj, u_proj_old, v_old, a_old)
    w_old.vector()[:] = w.vector()

    # increment time
    t += dt
    ii = ii + 1
    jj = jj + 1

    # Print progress of calculation
    # if ii%10 == 0:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(
        "Step: Sense   |   Simulation Time: {} s  |     Iterations: {}".format(
            t / 1e6, iter
        )
    )
    print()


# Final step paraview output
# file_results.write(u_v, t)
# file_results.write(omgPos_v, t)
# file_results.write(phi_v, t)
# file_results.write(omgNeg_v, t)
# file_results.write(Gshear_v, t)

# output final measured device voltage
#phi_top = w(x_plot, scaleY / 2, scaleZ / 2)[4] * phi_norm
#phi_btm = w(x_plot, -scaleY / 2.0, scaleZ / 2)[4] * phi_norm
#disp_out[ii] = w(scaleX, scaleY / 2 - slope * scaleY, scaleZ / 2)[0]
# voltage_out[ii] = phi_top - phi_btm
# charge_out[ii] = assemble(Farad*cMax*(cPos-cNeg)*dx(3))
# time_out[ii]    = t


