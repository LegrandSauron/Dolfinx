import numpy as np
import ufl

from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, plot
from dolfinx.io import gmshio

domain, cell_tags, facet_tags = gmshio.read_from_msh("t11_gmsh_malha.msh", MPI.COMM_SELF,0, gdim=2)

V = fem.VectorFunctionSpace(domain, ("CG", 2))


"""Condições de contorno"""
u_bc = np.array((0,) * domain.geometry.dim, dtype=PETSc.ScalarType)
fdim = domain.topology.dim - 1
dofs_2 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(101))
bcs=[fem.dirichletbc(u_bc,dofs=dofs_2,V=V)]

"""fim"""

B = fem.Constant(domain, PETSc.ScalarType((0, 0)))
T = fem.Constant(domain, PETSc.ScalarType((0, 0)))



v = ufl.TestFunction(V)
u = fem.Function(V)


# Spatial dimension
d = len(u)

# Identity tensor
I = ufl.variable(ufl.Identity(d))

# Deformation gradient
F = ufl.variable(I + ufl.grad(u))

# Right Cauchy-Green tensor
C = ufl.variable(F * F.T)

# Invariants of deformation tensors
Ic = ufl.variable(ufl.tr(C))
J  = ufl.variable(ufl.det(F))



# Elasticity parameters
E = PETSc.ScalarType(1.0e4)
nu = PETSc.ScalarType(0.3)
mu = fem.Constant(domain, E/(2*(1 + nu)))
lmbda = fem.Constant(domain, E*nu/((1 + nu)*(1 - 2*nu)))


# Stored strain energy density (compressible neo-Hookean model)
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2

# Stress
# Hyper-elasticity
P = ufl.diff(psi, F)
P = 2.0 * mu * ufl.sym(ufl.grad(u)) + lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * I

"""Tensor de hencky"""
#Hencky's Strain
#N, M = C.ufl_shape
#T_hencky=ufl.as_tensor([[(1/2)*(ufl.ln(C[i,j])) for i in range(N)] for j in range(M)])

"""Tensor de tensoes"""
#P= (1/J)*(2.0 * mu * T_hencky + lmbda * ufl.tr(T_hencky) * I )



metadata = {"quadrature_degree": 4}
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# Define form F (we want to find u such that F(u) = 0)
F = ufl.inner(ufl.grad(v), P)*dx - ufl.inner(v, B)*dx - ufl.inner(v, T)*ds(102) 


problem = fem.petsc.NonlinearProblem(F, u, bcs)


from dolfinx import nls
solver = nls.petsc.NewtonSolver(domain.comm, problem)

# Set Newton solver options
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"



import pyvista
import matplotlib.pyplot as plt
pyvista.start_xvfb()
plotter = pyvista.Plotter()
plotter.open_gif("deformation.gif", fps=3)

topology, cells, geometry = plot.create_vtk_mesh(u.function_space)
function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)

values = np.zeros((geometry.shape[0], 2))
values[:, :len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
function_grid["u"] = values
function_grid.set_active_vectors("u")

# Warp mesh by deformation
warped = function_grid.warp_by_vector("u", factor=0)
warped.set_active_vectors("u")

# Add mesh to plotter and visualize
actor = plotter.add_mesh(warped, show_edges=True, lighting=False, clim=[0, 10])

# Compute magnitude of displacement to visualize in GIF
Vs = fem.FunctionSpace(domain, ("Lagrange", 2))
magnitude = fem.Function(Vs)
us = fem.Expression(ufl.sqrt(sum([u[i]**2 for i in range(len(u))])), Vs.element.interpolation_points())
magnitude.interpolate(us)
warped["mag"] = magnitude.x.array



from dolfinx import log
log.set_log_level(log.LogLevel.INFO)
tval0 = -1.5
for n in range(1, 10):
    T.value[1] = n * tval0
    num_its, converged = solver.solve(u)
    assert(converged)
    u.x.scatter_forward()
    print(f"Time step {n}, Number of iterations {num_its}, Load {T.value}")
    function_grid["u"][:, :len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
    magnitude.interpolate(us)
    warped.set_active_scalars("mag")
    warped_n = function_grid.warp_by_vector(factor=1)
    plotter.update_coordinates(warped_n.points.copy(), render=False)
    plotter.update_scalar_bar_range([0, 10])
    plotter.update_scalars(magnitude.x.array)
    plotter.write_frame()
plotter.close()