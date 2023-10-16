"""FOrmulação semi-Discreta no tempo """

import dolfinx.fem
import dolfinx.mesh
import dolfinx.nls
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from matplotlib import pyplot as plt

#Mesh creation
nx = 1000     #number of nodes
#mesh = dolfinx.generation.IntervalMesh(MPI.COMM_WORLD, nx-1 ,[0,150e-6])
mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, nx-1 ,[0,150e-6] )
#Mixed function space creation
P1 = ufl.FiniteElement("CG", mesh.ufl_cell(), 2)
ME = dolfinx.fem.FunctionSpace(mesh,ufl.MixedElement([P1, P1]))

#Constant definition
D1 = dolfinx.fem.Constant(mesh, 8.5e-10)  
A = dolfinx.fem.Constant(mesh, 5.35e7)    
L = dolfinx.fem.Constant(mesh, 2.)         
l = dolfinx.fem.Constant(mesh, 5e-6)      
sigma = dolfinx.fem.Constant(mesh, 10.)    
alpha = dolfinx.fem.Constant(mesh, 2.94)  
c_le = dolfinx.fem.Constant(mesh, 0.03566433566433566)
c_se = dolfinx.fem.Constant(mesh, 1.)
alpha_eta = dolfinx.fem.Constant(mesh, 3.006406382595866e-06)
w_ = dolfinx.fem.Constant(mesh,2078893.9366884497)

#Function definition
def H(eta):
    return -2.*eta**3 + 3.*eta**2
def derH(eta):
    return 6.*(eta-eta**2)
def G(eta):
    return (1.-eta)**2*eta**2
def derG(eta):
    return 2*eta-6*eta**2+4*eta**3

#Boundary conditions
uD = dolfinx.fem.Function(ME)
uD.sub(0).interpolate(lambda x: 1 - x[0]/150e-6)
uD.sub(1).interpolate(lambda x: 1 - x[0]/150e-6)
uD.x.scatter_forward()
boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True, dtype=bool))
boundary_dofs = dolfinx.fem.locate_dofs_topological(ME, mesh.topology.dim - 1, boundary_facets)
bc = dolfinx.fem.dirichletbc(uD, boundary_dofs)

# Define test functions
q, v = ufl.TestFunctions(ME)
# Define functions
u = dolfinx.fem.Function(ME)
u0 = dolfinx.fem.Function(ME)
# Split mixed functions
c, eta = ufl.split(u)
c0, eta0 = ufl.split(u0)

# Define initial condition
u.sub(0).interpolate(lambda x: 1/(1+ufl.e**(2e6*(x[0]-0.90*150e-6))))
u.sub(1).interpolate(lambda x: 1/(1+ufl.e**(1.6e6*(x[0]-0.90*150e-6))))
u.x.scatter_forward()
u0.sub(0).interpolate(lambda x: 1/(1+ufl.e**(2e6*(x[0]-0.90*150e-6))))
u0.sub(1).interpolate(lambda x: 1/(1+ufl.e**(1.6e6*(x[0]-0.90*150e-6))))
u0.x.scatter_forward()

# Define temporal parameters
t = 0 # Start time
T = 100 # Final time
dt = 0.5 # Interval time
num_steps = int(T/dt)
k = dolfinx.fem.Constant(mesh, dt)

# Variational problem
F0 = ufl.inner(eta, v)*ufl.dx - ufl.inner(eta0, v)*ufl.dx + k*2*L*A*(c_le-1)*ufl.inner((c+c_le*(H(eta)-1.)-H(eta))*derH(eta), v)*ufl.dx + k*L*alpha_eta*ufl.inner(ufl.grad(eta), ufl.grad(v))*ufl.dx  + k*L*w_*ufl.inner(derG(eta),v)*ufl.dx

F1 = ufl.inner(c,q)*ufl.dx - ufl.inner(c0,q)*ufl.dx + k*ufl.inner(D1*ufl.grad(c), ufl.grad(q))*ufl.dx - k*(1-c_le)*derH(eta)*ufl.inner(D1*ufl.grad(eta), ufl.grad(q))*ufl.dx

F = F0 + F1

from dolfinx import nls

problem = dolfinx.fem.petsc.NonlinearProblem(F, u, bcs=[bc])
solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
solver.atol = 1e-12
solver.rtol = 1e-12
solver.convergence_criterion = "incremental"
solver.report = True
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

from dolfinx.io import XDMFFile
file = XDMFFile(MPI.COMM_WORLD, "resultados/time_integration_nolinear_couple.xdmf", "w")
file.write_mesh(mesh)


tol = 1e-8
while (t < (T-tol)):
    t += dt
    r = solver.solve(u)
    u.x.scatter_forward()

    if t < (T-tol): #To keep u0 in last step
        u.vector.copy(result=u0.vector)
    file.write_function(u.sub(1), t)

