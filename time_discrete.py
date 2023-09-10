from mpi4py import MPI
from petsc4py import PETSc
import numpy as np 
from dolfinx import fem, mesh, io, plot
from ufl import*


"""Utilizando apenas a estrutura de codigo para a construção da discretização no tempo e do tensor de piola."""

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



matInd = fem.Constant(domain,1.)
Gshear = fem.Constant(domain,0.003e-6) 
Kbulk = fem.Constant(domain,2000*0.003e-6)
Gshear0 = 100.0e-6 
Im_gent = fem.Constant(domain,300.)

# Mass density
densidade_massa = fem.Constant(domain,1e-9)  # 1e3 kg/m^3 = 1e-9 mg/um^3,

"""Constantes do metodo alpha generalizado """
# Generalized-alpha method parameters
alpha = fem.Constant(domain, 0.0)
gamma   = fem.Constant(domain, 0.5)
beta    = fem.Constant(domain,(0.5+0.5)**2./4.)

"""Valores de tempo"""
# Define temporal parameters
#Simulation time related params (reminder: in microseconds)
ttd  = 0.01
# Step in time
t = 0.0         # initialization of time
T_tot = 0e6*1.0e6 #200.0 #0.11        # total simulation time 

t1 = 15.0e6
t2 = 35.0e6
t3 = 2.5e6
t4 = 52.5e6
T2_tot = 30.0e6 #0.0001*1.e6 #t1+t2+t3+t4
dt = T2_tot/500


#constants
carregamento= 50000
E = 78e6
poisson = 0.3
lambda_ = E*poisson / ((1+poisson)*(1-2*poisson))
G= E / 2*(1+poisson)



# Define function space, scalar
U2=fem.VectorFunctionSpace(domain, ("CG", 1))

# DOFs

#TH = MixedElement([U2])
#ME = ufl.FunctionSpace(domain, TH) # Total space for all DOFs
#ME= MixedFunctionSpace([U2])
ME= fem.VectorFunctionSpace(domain, ("CG", 1))
  # Scalar space for visualization later


# Define test functions in weak form
dw = TrialFunction(ME)                                   
(u_test)  = TestFunctions(ME)    # Test function

# Define actual functions with the required DOFs
w = fem.Function(ME)
(u) = split(w)    # current DOFs

# A copy of functions to store values in last step for time-stepping.
w_old = fem.Function(ME)
(u_old) = split(w_old)   # old DOFs

v_old = fem.Function(ME)
a_old = fem.Function(ME)

# Initial chemical potential

def F_calc(u):
    dim = len(u)
    Id = Identity(dim) # Identity tensor
    
    F = Id + grad(u) # 3D Deformation gradient
    return F # Full 3D F

    
def Piola(F):
    
    Id = Identity(3)
    J = det(F)
    
    C = F.T*F
    Cdis = J**(-2/3)*C  
    I1 = tr(Cdis)
  
    # Piola stress (Gent)
    TR = J**(-2/3)*G*(Im_gent/(Im_gent+3-I1))*(F - 1/3*tr(Cdis)*inv(F.T))\
        + Kbulk*ln(J)*inv(F.T) 
    
    return TR



# variable time step
dk = fem.Constant(domain,0.0)

"""funções de aceleração, velocidade e campo"""
# Update formula for acceleration
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_a(u, u_old, v_old, a_old):
    
    dt_ = dk
    beta_ = beta
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old


# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, u_old, v_old, a_old):
    
    dt_ = dk
    gamma_ = gamma
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

def update_fields(u_proj, u_proj_old, v_old, a_old):
    """Update fields at the end of each time step."""

    # Get vectors (references)
    u_vec, u0_vec  = u_proj.vector(), u_proj_old.vector()
    v0_vec, a0_vec = v_old.vector(), a_old.vector()

    # use update functions using vector arguments
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec)

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
a_new = update_a(u, u_old, v_old, a_old)
v_new = update_v(a_new, u_old, v_old, a_old)

# get avg fields for generalized-alpha method
u_avg = avg(u_old, u, alpha)
v_avg = avg(v_old, v_new, alpha)



# Kinematics
F = F_calc(u_avg)
C = F.T*F
Ci = inv(C)
F_old = F_calc(u_old) #grad(u_old) + Identity(3)
J = det(F)
J_old = det(F_old)

'''''''''''''''''''''''
       WEAK FORMS
'''''''''''''''''''''''
dynSwitch = fem.Constant(1.0)
     
L0 = inner(Piola(F_calc(u_avg)), grad(u_test))*dx + dynSwitch*densidade_massa*inner(a_new, u_test)*dx 
   








"""

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
    prm['nonlinear_solver'] = 'newton'
    prm['newton_solver']['linear_solver'] = 'petsc'   #'petsc'   #'gmres'
    prm['newton_solver']['absolute_tolerance'] = 1.E-6    # 1.e-10
    prm['newton_solver']['relative_tolerance'] = 1.E-6    # 1.e-10
    prm['newton_solver']['maximum_iterations'] = 60
    
    
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
 
"""

