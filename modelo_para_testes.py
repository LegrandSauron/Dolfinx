

# Fenics-related packages

from dolfinx import fem, mesh,log,nls, cpp
import numpy as np
import math
# Plotting packages
import matplotlib.pyplot as plt
# Current time package
from datetime import datetime
from sympy import dirichlet_eta
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
                
M1 = 60 #N/scaleY*int1
M2 = 2.0 #90 #int2/scaleY*N
M3 = M1/2
r1 = 1/1.5
r2 = 1/1.06     
r3 = r1
a1 = (1-r1)/(1-r1**M1)
a2 = (1-r2)/(1-r2**(M2)) 
a3 = (1-r3)/(1-r3**M3)
                
preMapLength = float(int1 + 2*M2*(int1/M1))

domain =mesh.create_box((0.,-preMapLength, 0.0),(scaleX, preMapLength, scaleZ),xElem,N, zElem)
#mesh = BoxMesh(Point(0.,-scaleY/2,0.),Point(scaleX, scaleY/2, scaleZ),xElem,N,zElem)


xOrig = domain.coordinates()
xMap1 = np.zeros((len(xOrig),3))
xMap2 = np.zeros((len(xOrig),3))
xMap3 = np.zeros((len(xOrig),3))

slope = 0.00

domain.coordinates()[:] = xMap3

x = ufl.SpatialCoordinate(domain) 

#x = mesh.coordinates()

#x[:,0] *= scaleX
#x[:,1] *= scaleY
#x[:,2] *= scaleZ

#################### ##################################################
"""Principais problemas :
"""
class SubDomain():
    pass

def near():
    pass

def project():
    pass

def MeshFunction():
    return
# Dirichlet boundary
# Mark boundary subdomians

"""facets: """



tol=1e-5
tol = 1.e-5


"""
Userdefined expression for defining different materials
"""
class UserExpression():
    pass

# Rayleigh damping coefficients
eta_m = fem.Constant(0.00000) # Constant(0.000005)
eta_k = fem.Constant(0.00000) # Constant(0.000005)


"""Constantes do metodo alpha generalizado """
# Generalized-alpha method parameters
alpha = fem.Constant(0.0)
gamma   = fem.Constant(0.5+alpha)
beta    = fem.Constant((gamma+0.5)**2/4.)


#Simulation time related params (reminder: in microseconds)
ttd  = 0.01
# Step in time
t = 0.0         # initialization of time
T_tot = 0e6*1.0e6 #200.0 #0.11        # total simulation time 

#

t1 = 15.0e6
t2 = 35.0e6
t3 = 2.5e6
t4 = 52.5e6
T2_tot = 30.0e6 #0.0001*1.e6 #t1+t2+t3+t4
dt = T2_tot/500

# Define function space, scalar
U2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
D0 = ufl.FiniteElement("DG", domain.ufl_cell(), 0)
D1 = ufl.FiniteElement("DG", domain.ufl_cell(), 1)


# DOFs
TH = ufl.MixedElement([U2, P1, P1, P1])
#ME = ufl.FunctionSpace(domain, TH) # Total space for all DOFs
ME= ufl.MixedFunctionSpace([U2, P1, P1, P1])

W = ufl.FunctionSpace(domain,P1)   # Scalar space for visualization later
W2 = ufl.FunctionSpace(domain,U2)   # Vector space for visualization later
W3 = ufl.FunctionSpace(domain,D0)   # DG space for visualization later
W4 = ufl.FunctionSpace(domain,D1)   # DG space for visualization later

# Define test functions in weak form
dw = ufl.TrialFunction(ME)                                   
(u_test, omgPos_test, phi_test, omgNeg_test)  = ufl.TestFunctions(ME)    # Test function

# Define actual functions with the required DOFs
w = fem.Function(ME)
(u, omgPos, phi, omgNeg) = ufl.split(w)    # current DOFs

# A copy of functions to store values in last step for time-stepping.
w_old = fem.Function(ME)
(u_old, omgPos_old, phi_old, omgNeg_old) = ufl.split(w_old)   # old DOFs

v_old = fem.Function(W2)
a_old = fem.Function(W2)

# Initial chemical potential
mu0 = ufl.ln(Concentracao_positiva)
mu20 = ufl.ln(concentracao_negativa)




init_omgPos = fem.Expression('abs(x[1])>=int2-tol && abs(x[1])<=scaleY/2+tol?std::log((cPos0)):std::log((cNum))', int2=int2, scaleY=scaleY, tol = tol, cPos0 = Concentracao_positiva, cNum=DOLFIN_EPS, degree=0)
omgPos_init =interpolate(init_omgPos,ME.sub(1).collapse())
assign(w_old.sub(1),omgPos_init)


init_omgNeg = fem.Expression('abs(x[1])>=int2-tol && abs(x[1])<=scaleY/2+tol?std::log((cNeg0)):std::log((cNum))', int2=int2, scaleY=scaleY, tol = tol, cNeg0 = concentracao_negativa,  cNum=DOLFIN_EPS, degree=0)
omgNeg_init = interpolate(init_omgNeg,ME.sub(3).collapse())

fem.assign(w_old.sub(3),omgNeg_init)

# Update initial guess for w
fem.assign(w.sub(3),omgNeg_init)
assign(w.sub(1),omgPos_init)
#assign(w.sub(5),cNeg_init)
#assign(w.sub(2),cPos_init)

cPos     = ufl.exp(omgPos - Farad*phi*phi_norm/RT)
cNeg     = ufl.exp(omgNeg + Farad*phi*phi_norm/RT)
cPos_old = ufl.exp(omgNeg_old + Farad*phi_old*phi_norm/RT)

#Ny = N+1
#y = np.linspace(0, scaleY, Ny)
y = np.sort(np.array(domain.coordinates()[:,1])) #np.linspace(-scaleY/2, scaleY/2, Ny)
#f = 2/0

# Quick-calculate sub-routines
carregamento= 50000
E = 78e6
poisson = 0.3
lambda_ = E*poisson / ((1+poisson)*(1-2*poisson))
G= E / 2*(1+poisson)



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
    T_max = vareps0*vareps_r*J*(ufl.outer(e_sp,e_sp) - 1/2*(ufl.inner(e_sp,e_sp))*Id)*ufl.inv(F.T) 
    
    # Piola stress (Gent)
    TR = J**(-2/3)*Gshear*(Im_gent/(Im_gent+3-I1))*(F - 1/3*ufl.tr(Cdis)*ufl.inv(F.T))\
        + Kbulk*ufl.ln(J)*ufl.inv(F.T) + T_max
    
    return TR

# variable time step
dk = fem.Constant(0.0)

"""funções de aceleração, velocidade e campo"""
# Update formula for acceleration
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dk
        beta_ = beta
    else:
        dt_ = float(dk)
        beta_ = float(beta)
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old


# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dk
        gamma_ = gamma
    else:
        dt_ = float(dk)
        gamma_ = float(gamma)
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

def update_fields(u_proj, u_proj_old, v_old, a_old):
    """Update fields at the end of each time step."""

    # Get vectors (references)
    u_vec, u0_vec  = u_proj.vector(), u_proj_old.vector()
    v0_vec, a0_vec = v_old.vector(), a_old.vector()

    # use update functions using vector arguments
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

    # Update (u_old <- u)
    v_old.vector()[:], a_old.vector()[:] = v_vec, a_vec
    #u_old.vector()[:] = u_proj.vector()

def ppos(x):
    return (x+abs(x))/2.

def avg(x_old, x_new, alpha):
    return alpha*x_old + (1-alpha)*x_new

'''''''''''''''''''''''''''''''''''''''''
  KINEMATICS & CONSTITUTIVE RELATIONS
'''''''''''''''''''''''''''''''''''''''''

# Newmark-beta kinematical update
a_new = update_a(u, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

# get avg fields for generalized-alpha method
u_avg = avg(u_old, u, alpha)
v_avg = avg(v_old, v_new, alpha)
omgPos_avg = avg(omgPos_old, omgPos, alpha)
phi_avg    = avg(phi_old, phi, alpha)
omgNeg_avg = avg(omgNeg_old, omgNeg, alpha)

# Explicit concentration updates
cPos     = ufl.exp(omgPos_avg - phi_avg)
cNeg     = ufl.exp(omgNeg_avg + phi_avg)
cPos_old = ufl.exp(omgPos_old - phi_old)
cNeg_old = ufl.exp(omgNeg_old + phi_old)


# Kinematics
F = F_calc(u_avg)
C = F.T*F
Ci = ufl.inv(C)
F_old = F_calc(u_old) #grad(u_old) + Identity(3)
J = ufl.det(F)
J_old = ufl.det(F_old)


'''''''''''''''''''''''
       WEAK FORMS
'''''''''''''''''''''''

dynSwitch = fem.Constant(1.0)
     
L0 = ufl.inner(Piola(F_calc(u_avg), phi_avg), ufl.grad(u_test))*dx \
         + dynSwitch*densidade_massa*ufl.inner(a_new, u_test)*dx 
    
L1 = ufl.dot((cPos-cPos_old)/dk/D,omgPos_test)*dx \
    + (cPos_old)*ufl.inner(Ci*matInd*ufl.grad(omgPos_avg),ufl.grad(omgPos_test))*dx
   
L3 = ufl.dot(vareps*phi_norm*J*Ci*ufl.grad(phi_avg),ufl.grad(phi_test))*dx \
     -ufl.dot(Farad*matInd*concentracao_maxima*(cPos-cNeg),phi_test)*dx 

L4 = ufl.dot((cNeg-cNeg_old)/dk/D,omgNeg_test)*dx \
    + (cNeg_old)*ufl.dot(Ci*matInd*ufl.grad(omgNeg_avg),ufl.grad(omgNeg_test))*dx
  

# Total weak form

L = (1/Gshear0)*L0 + L1 + L3 + L4 
"""
L = L0 + L1 + L2 + L3
"""
# Automatic differentiation tangent:
a = ufl.derivative(L, w, dw)


# Boundary condition expressions as necessary

disp = fem.Expression(("0.5*t/Tramp"),
                  Tramp = T2_tot-T_tot, t = 0.0, degree=1)
disp2 = fem.Expression(("-0.5*t/Tramp"),
                  Tramp = T2_tot-T_tot, t = 0.0, degree=1)

phiRamp = fem.Expression(("(250/phi_norm)*t/Tramp"),
                  phi_norm = phi_norm, Tramp = T_tot, t = 0.0, degree=1)

# Boundary condition definitions

bcs_1 = fem.dirichletbc(ME.sub(2), 0., facets, 8) # Ground center of device
bcs_2 = fem.dirichletbc(ME.sub(2), 0., facets, 6) # Ground top of device
bcs_3 = fem.dirichletbc(ME.sub(2), 0., facets, 3) # Ground bottom of device

bcs_4 = fem.dirichletbc(ME.sub(2), phiRamp, facets, 6) # Ground bottom of device

bcs_a = fem.dirichletbc(ME.sub(0).sub(0),0.,facets,2)


x_plot = scaleX
phi_eq = w_old(x_plot, scaleY/2, scaleZ/2)[4]
#phiRamp = Expression(("phi_eq + 2.0e3/phi_norm*(1-exp(-10*t/Tramp)) + 1.e3/phi_norm*sin(2*pi*f*t/1e6)"),
#                 phi_eq=phi_eq, phi_norm = phi_norm, pi=np.pi, f=100,  Tramp = T2_tot, t = 0.0, degree=1)
phiRamp = fem.Expression(("min(1.0e3/phi_norm*(t/Tramp), 1.0e3/phi_norm)"),
                 phi_eq=phi_eq, phi_norm = phi_norm, pi=np.pi, Tramp = T2_tot, t = 0.0, degree=2)
disp = fem.Expression(("5.5*scaleX*t/Tramp"),
                  scaleX = scaleX, Tramp = T2_tot, t = 0.0, degree=1)\
    
bcs_f = fem.dirichletbc(ME.sub(2),phiRamp,facets,6) # Ramp up phi on top face

#bcs_b1 = DirichletBC(ME.sub(0),Constant((0.,0., 0.)),facets,2) # left face built-in
bcs_b1 = fem.dirichletbc(ME.sub(0).sub(0),0.0,facets,2) # pull right face
bcs_b2 = fem.dirichletbc(ME.sub(0).sub(0),disp,facets,7) # pull right face
bcs_b3 = fem.dirichletbc(ME.sub(0).sub(1),0.0,facets,10)
bcs_b3a = fem.dirichletbc(ME.sub(0).sub(0),0.0,facets,10)
bcs_b3b = fem.dirichletbc(ME.sub(0).sub(2),0.0,facets,10)
bcs_b3c = fem.dirichletbc(ME.sub(0).sub(0),0.0,facets,9)
bcs_b3d = fem.dirichletbc(ME.sub(0).sub(2),0.0,facets,9)
bcs_b4 = fem.dirichletbc(ME.sub(0).sub(2),0.0,facets,5)


#bcs2 = [bcs_3, bcs_a, bcs_b1, bcs_b3, bcs_b3a, bcs_b3b, bcs_b3c, bcs_b3d, bcs_b4, bcs_f]

bcs2 = [bcs_3, bcs_a, bcs_b1, bcs_b3, bcs_b4, bcs_f]

from dolfinx.io import XDMFFile
# Output file setup
file_results = XDMFFile("results/suo_capacitor_3D_traction.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True



# initialize counter
ii = 0
'''''''''''''''''''''
    RUN ANALYSIS
'''''''''''''''''''''
u_v = w_old.sub(0)
u_v.rename("disp","")

omgPos_v = w_old.sub(1)
omgPos_v.rename("omega", "")

phi_v = w_old.sub(2)
phi_v.rename("phi", "")

omgNeg_v = w_old.sub(3)
omgNeg_v.rename("omega2", "")

Gshear_v = project(1e6*Gshear,W3)
Gshear_v.rename("Gshear","")
file_results.write(Gshear_v,t)

Kbulk_v = project(1e6*Kbulk,W3)
Kbulk_v.rename("Kbulk","")
file_results.write(Kbulk_v,t)

intInd_v = project(intInd,W3)
intInd_v.rename("IntIndex","")
file_results.write(intInd_v,t)



# Initialize arrays for plotting later
Ny = N+1
x_plot = scaleX
y = np.sort(np.array(domain.coordinates()[np.where(domain.coordinates()[:,0] == x_plot),1])) #np.linspace(-scaleY/2, scaleY/2, Ny)
y = y[0,:]
y2 = np.linspace(0, scaleY/10, Ny) 

output_step = [0.,0.001,0.005,5.01, 50.01]
op_dim = len(output_step)
phi_var = np.zeros((Ny, op_dim))
cPos_var = np.zeros((Ny, op_dim))
#cNeg_var = np.zeros((Ny, op_dim))

voltage_out = np.zeros(100000)
charge_out = np.zeros(100000)
disp_out = np.zeros(100000)
time_out    = np.zeros(100000)
trac_out    = np.zeros(100000)

i=0
ii=0

trac_out    = np.zeros(100000)
ocv_out    = np.zeros(100000)
t_out    = np.zeros(100000)
t = T_tot
jj = 0
spike = 0

# Turn on implicit dynamics
dynSwitch.assign(fem.Constant(1.0))

voltage_out = np.zeros(100000)
charge_out = np.zeros(100000)
disp_out = np.zeros(100000)
time_out    = np.zeros(100000)
trac_out    = np.zeros(100000)

while (round(t,2) <= round(T_tot + 2.0*T2_tot,2)):
    
    
    # Output storage
    #if ii%10 == 0:
    if True:
        file_results.write(u_v,t)
        file_results.write(omgPos_v, t)
        file_results.write(phi_v, t)
        phi_nonDim = phi_norm/1.e3*phi
        phi_nonDim = project(phi_nonDim,W)
        phi_nonDim.rename("phi_Orig","")
        file_results.write(phi_nonDim, t)
        file_results.write(omgNeg_v, t)
        file_results.write(Gshear_v,t)
        
        #
        _w_0,_w_1, _w_2, _w_3 = w_old.split()
   
    dt = T2_tot/20
    
    
    #dt = T2_tot/1000
    dk.assign(dt)
    
    #disp2.t = min(t - T_tot, T2_tot)
    phiRamp.t = t - T_tot - float(alpha*dt)
    
    trac_final = 0.040e-6 # 40 kPa total
    n = ufl.FacetNormal(domain)
    
    if t-T_tot<=T2_tot:
        trac_value = 0.0
    else:
        trac_value = trac_final*(t - T_tot - T2_tot - float(alpha*dt))/T2_tot
    
    trac = -trac_value*n
    trac_out[jj] = trac_value
       
    Ltrac = -ufl.dot(trac/Gshear0, u_test)*ds(9) # + dot(trac/Gshear0, u_test)*ds(3)
    # Set up the non-linear problem (free swell)
    StressProblem = fem.petsc.NonlinearProblem(L+Ltrac, w, bcs2, J=a)
    
    # Set up the non-linear solver
    solver  = nls.petsc.NewtonSolver(StressProblem)
    # Solver parameters
    
    prm = solver.parameters

    # Solve the problem
    (iter, converged) = solver.solve()
    
    
    # output measured device voltage, etc.
    phi_top = w(x_plot, scaleY/2, scaleZ/2)[4]*phi_norm
    phi_btm = w(x_plot, -scaleY/2, scaleZ/2)[4]*phi_norm
    voltage_out[ii] = phi_top - phi_btm
    disp_out[ii]    = w(scaleX, scaleY/2 - slope*scaleY, scaleZ/2)[0]
    time_out[ii]    = t - float(alpha*dt)
    #trac_out[jj]    = trac_final*(t-T_tot)/(T2_tot)
    ocv_out[jj] = phi_top - phi_btm
    t_out[jj]   = t - T_tot -float(alpha*dt)
    #
    charge_out[ii] = assemble(Farad*concentracao_maxima*(cPos-cNeg)*dx(3))
    
    
    u_proj = project(u, W2)
    u_proj_old = project(u_old, W2)
    update_fields(u_proj, u_proj_old, v_old, a_old)
    w_old.vector()[:] = w.vector()
    
    # increment time
    t += dt
    ii = ii + 1
    jj = jj + 1
    
    
    # Print progress of calculation
    #if ii%10 == 0:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Step: Sense   |   Simulation Time: {} s  |     Iterations: {}".format(t/1e6, iter))
    print()
    



