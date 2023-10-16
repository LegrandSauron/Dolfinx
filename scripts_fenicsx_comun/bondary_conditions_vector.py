
import numpy as np

import ufl
from dolfinx import fem, nls, mesh, plot, log, io
from mpi4py import MPI
from petsc4py import PETSc

import pdb


# ----------------------------------------------------------------------------------------------------------------------
# Create a box mesh of a beam
L = 20.0
H = 1.0
msh = mesh.create_box(MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]), np.array([L, H, H])], [10, 1, 1],
                      mesh.CellType.hexahedron)
P1 = ufl.VectorElement("CG", msh.ufl_cell(), 1, dim=3)  # Lagrange family, 1st order
P2 = ufl.VectorElement("CG", msh.ufl_cell(), 1, dim=3)  # Lagrange family, 1st order
VY = fem.FunctionSpace(msh, ufl.MixedElement([P1, P2]))


# ----------------------------------------------------------------------------------------------------------------------
# Boundary conditions and body loads
def left(x):
    return np.isclose(x[0], 0)

def right(x):
    return np.isclose(x[0], L)

def top(x):
    return np.isclose(x[1], 1)

def bottom(x):
    return np.isclose(x[1], 0)


def fixed_displacement_expression(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1])))

def incremented_displacement_expression(x):
    return np.full(x.shape[1], 1.0e-02)

def fixed_director_expression(x):
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1])))

# ----------------------------------------------
# Dirichlet boundary application to affected displacement dofs

# Fix displacement DOF on left face and apply axial displacement on right face
V0, submap = VY.sub(0).collapse()
V0x, subsubmap = V0.sub(0).collapse()
fixed_displacement = fem.Function(V0)
fixed_displacement.interpolate(fixed_displacement_expression)
incremented_displacement = fem.Function(V0x)
incremented_displacement.interpolate(incremented_displacement_expression)
left_u_dofs = fem.locate_dofs_geometrical((VY.sub(0),V0), left)
right_u_dofs = fem.locate_dofs_geometrical((VY.sub(0).sub(0),V0x), right)
bc1 = fem.dirichletbc(fixed_displacement, left_u_dofs, VY.sub(0))
bc2 = fem.dirichletbc(incremented_displacement, right_u_dofs, VY.sub(0).sub(0))

# Fix ALL change of director degrees of freedom in the domain
V1, submap = VY.sub(1).collapse()
fixed_director = fem.Function(V1)
fixed_director.interpolate(fixed_director_expression)
top_wf_dofs = fem.locate_dofs_geometrical((VY.sub(1),V1), top)
bottom_wf_dofs = fem.locate_dofs_geometrical((VY.sub(1),V1), bottom)
bc3 = fem.dirichletbc(fixed_director, top_wf_dofs, VY.sub(1))
bc4 = fem.dirichletbc(fixed_director, bottom_wf_dofs,VY.sub(1))

bcs = [bc1,  bc2, bc3, bc4]

# ----------------------------------------------------------------------------------------------------------------------
# Function space
v, yf = ufl.TestFunctions(VY)     # test functions
uwf = fem.Function(VY)            # current displacement and change of director
u, wf = ufl.split(uwf)

# ----------------------------------------------------------------------------------------------------------------------
# Kinematics

# Spatial dimension
d = len(u)

# Identity tensor
I = ufl.variable(ufl.Identity(d))

# Deformation gradient
F = ufl.variable(I + ufl.grad(u))

# Right Cauchy-Green tensor
C = ufl.variable(F.T * F)

# Invariants
Ic = ufl.variable(ufl.tr(C))
J  = ufl.variable(ufl.det(F))

# ----------------------------------------------------------------------------------------------------------------------
# Constitutive law

# Set the elasticity parameters
E = PETSc.ScalarType(1.0e4)
nu = PETSc.ScalarType(0.3)
mu = fem.Constant(msh, E/(2*(1 + nu)))
lmbda = fem.Constant(msh, E*nu/((1 + nu)*(1 - 2*nu)))

# Stored strain energy density (compressible neo-Hookean model)
psi_m = (mu / 2.0) * (Ic - 3.0) - mu * ufl.ln(J) + (lmbda / 2.0) * (ufl.ln(J))**2

# 1st Piola Kirchhoff stress tensor
P = ufl.diff(psi_m, F)                                     # 1st Piola Kirchhoff stress tensor
df = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0, 0.0)))  # dummy director stress

# ----------------------------------------------------------------------------------------------------------------------
# Variational formulation
metadata = {"quadrature_degree": 3}
dx = ufl.Measure("dx", domain=msh, metadata=metadata)

# Define form Pi (we want to find u such that Pi(u) = 0)
Pi = ufl.inner(P, ufl.grad(v))*dx + ufl.inner(df, yf)*dx

# Initialize nonlinear problem
problem = fem.petsc.NonlinearProblem(Pi, uwf, bcs)

# ----------------------------------------------------------------------------------------------------------------------
# Initialise Newton Rhapson solver
solver = nls.petsc.NewtonSolver(msh.comm, problem)

# Set Newton solver options
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"

# ----------------------------------------------------------------------------------------------------------------------
# Incremental application of the traction loading
log.set_log_level(log.LogLevel.INFO)
for n in range(1, 2):
    num_its, converged = solver.solve(uwf)
    assert(converged)

with io.XDMFFile(MPI.COMM_WORLD, "resultados/u.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(uwf.split()[0])