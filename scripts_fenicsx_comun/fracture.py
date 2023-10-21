import gmsh
import numpy as np
from mpi4py import MPI
import csv
import pandas as pd

import ufl
import time
from dolfinx import fem, io, mesh, plot, common, geometry
from ufl import ds, dx, grad, inner
import pyvista
from dolfinx.plot import create_vtk_mesh
from dolfinx.fem.petsc import LinearProblem

from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from dolfinx.io import gmshio

teststart = time.perf_counter()
print("start the simulation")

rank = MPI.COMM_WORLD.rank
comm = MPI.COMM_WORLD

W = 1
H = 0.0001
gdim = 2

crack_marker = 10

gmsh.initialize()

if rank == 0:
    box = gmsh.model.occ.addRectangle(0, 0, 0, W, W)
    rectangle1 = gmsh.model.occ.addRectangle(0, W / 2 - H, 0, W / 2, H)

    whole_domain = gmsh.model.occ.cut([(2, box)], [(2, rectangle1)])
    gmsh.model.occ.synchronize()

    material_marker = 1
    volumes = gmsh.model.getEntities(dim=gdim)
    assert len(volumes) == 1
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], material_marker)  # box
    gmsh.model.setPhysicalName(volumes[0][0], material_marker, "materials")

    # crack_marker = 10
    crack = []

    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        # print(boundary, center_of_mass)
        if np.allclose(center_of_mass, [0.5, 0.49995, 0]):
            crack.append(boundary[1])
        if np.allclose(center_of_mass, [0.25, 0.4999, 0]):
            crack.append(boundary[1])
        if np.allclose(center_of_mass, [0.25, 0.5, 0]):
            crack.append(boundary[1])

    gmsh.model.addPhysicalGroup(1, crack, crack_marker)
    gmsh.model.setPhysicalName(1, crack_marker, "crack_wall")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.008)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.008)

    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.model.mesh.generate(gdim)
    gmsh.write("mesh_applied.msh")

gmsh.finalize()

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
msh, cell_markers, facet_markers = gmshio.read_from_msh("mesh_applied.msh", mesh_comm, gmsh_model_rank, gdim=gdim)
# print(facet_markers.find(crack_marker))
V = fem.FunctionSpace(msh, ("CG", 1))
W = fem.VectorFunctionSpace(msh, ("CG", 1))
WW = fem.FunctionSpace(msh, ("DG", 0))

p, q = ufl.TrialFunction(V), ufl.TestFunction(V)
u, v = ufl.TrialFunction(W), ufl.TestFunction(W)

Gc = 2.7
l = 0.015
lmbda = 121.15e3
mu = 80.77e32


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


def psi(u):
    return 0.5 * (lmbda + mu) * (
        0.5 * (ufl.tr(epsilon(u)) + abs(ufl.tr(epsilon(u))))
    ) ** 2 + mu * inner(ufl.dev(epsilon(u)), ufl.dev(epsilon(u)))


def H(uold, unew, Hold):
    return ufl.conditional(ufl.lt(psi(uold), psi(unew)), psi(unew), Hold)


def bottom(x):
    return np.isclose(x[1], 0)


bottom_sld_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, bottom)
bottom_sld_dofs_x = fem.locate_dofs_topological(W.sub(0), msh.topology.dim - 1, bottom_sld_facets)
bottom_sld_bc_x = fem.dirichletbc(ScalarType(0), bottom_sld_dofs_x, W.sub(0))
bottom_sld_dofs_y = fem.locate_dofs_topological(W.sub(1), msh.topology.dim - 1, bottom_sld_facets)
bottom_sld_bc_y = fem.dirichletbc(ScalarType(0), bottom_sld_dofs_y, W.sub(1))


def top(x):
    return np.isclose(x[1], 1)


top_sld_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, top)
top_sld_dofs_x = fem.locate_dofs_topological(W.sub(0), msh.topology.dim - 1, top_sld_facets)
top_sld_bc_x = fem.dirichletbc(ScalarType(0), top_sld_dofs_x, W.sub(0))
top_sld_dofs_y = fem.locate_dofs_topological(W.sub(1), msh.topology.dim - 1, top_sld_facets)

load = 0.0
top_y_load = fem.dirichletbc(load, top_sld_dofs_y, W.sub(1))


bc_u = [bottom_sld_bc_x, bottom_sld_bc_y, top_y_load]

f = fem.Constant(msh, ScalarType((0, 0)))
T = fem.Constant(msh, ScalarType((0, 0)))
ds = ufl.Measure("ds", domain=msh)

fdim = msh.topology.dim - 1
initial_crack = fem.Function(V)
initial_crack.interpolate(lambda x: 1.0 + 0 * x[0])
bc_phi = [fem.dirichletbc(initial_crack, fem.locate_dofs_topological(V, fdim, facet_markers.find(crack_marker)))]

unew, uold = fem.Function(W), fem.Function(W)
pnew, pold, Hold = fem.Function(V), fem.Function(V), fem.Function(V)

E_du = ((1.0 - pold) ** 2) * inner(grad(v), sigma(u)) * dx
E_phi = (Gc * l * inner(grad(p), grad(q)) + ((Gc / l) + 2.0 * H(uold, unew, Hold)) * inner(p, q) - 2.0 * H(uold, unew, Hold) * q) * dx

a_sld = ufl.lhs(E_du)
L_sld = ufl.rhs(E_du) + ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds
a_E_phi = ufl.lhs(E_phi)
L_E_phi = ufl.rhs(E_phi)

bilinear_form_sld = fem.form(a_sld)
linear_form_sld = fem.form(L_sld)
bilinear_form_E_phi = fem.form(a_E_phi)
linear_form_E_phi = fem.form(L_E_phi)

A_sld = fem.petsc.create_matrix(bilinear_form_sld)
b_sld = fem.petsc.create_vector(linear_form_sld)
A_E_phi = fem.petsc.create_matrix(bilinear_form_E_phi)
b_E_phi = fem.petsc.create_vector(linear_form_E_phi)

solver_sld = PETSc.KSP().create(msh.comm)
solver_sld.setOperators(A_sld)
solver_sld.setType(PETSc.KSP.Type.GMRES)
solver_sld.getPC().setType(PETSc.PC.Type.JACOBI)  

solver_E_phi = PETSc.KSP().create(msh.comm)
solver_E_phi.setOperators(A_E_phi)
solver_E_phi.setType(PETSc.KSP.Type.GMRES)
solver_E_phi.getPC().setType(PETSc.PC.Type.JACOBI)

t = 0
u_r = 0.007
deltaT = 0.1
tol = 1e-4

while t <= 1.0:
    t += deltaT
    
    if t >= 0.7:
        deltaT = 0.0001

    load = t * u_r
    top_y_load = fem.dirichletbc(load, top_sld_dofs_y, W.sub(1))
    bc_u = [bottom_sld_bc_x, bottom_sld_bc_y, top_y_load]

    iter = 0
    err = 1

    while err > tol:
        iter += 1

        with b_sld.localForm() as loc_b_sld:
            loc_b_sld.set(0)
        A_sld.zeroEntries()
        fem.petsc.assemble_matrix(A_sld, bilinear_form_sld, bcs=bc_u)
        A_sld.assemble()
        fem.petsc.assemble_vector(b_sld, linear_form_sld)   
        fem.petsc.apply_lifting(b_sld, [bilinear_form_sld], [bc_u])
        b_sld.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b_sld, bc_u)

        solver_sld.solve(b_sld, unew.vector)
        unew.x.scatter_forward()

        with b_E_phi.localForm() as loc_b_E_phi:
            loc_b_E_phi.set(0)
        A_E_phi.zeroEntries()
        fem.petsc.assemble_matrix(A_E_phi, bilinear_form_E_phi, bcs=bc_phi)
        A_E_phi.assemble()
        fem.petsc.assemble_vector(b_E_phi, linear_form_E_phi)   
        fem.petsc.apply_lifting(b_E_phi, [bilinear_form_E_phi], [bc_phi])
        b_sld.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b_E_phi, bc_phi)

        solver_E_phi.solve(b_E_phi, pnew.vector)
        pnew.x.scatter_forward()

        error_form_p = fem.form((pnew - pold) ** 2 * ufl.dx)
        error_form_u = fem.form((unew - uold) ** 2 * ufl.dx)
        error_p = np.sqrt(comm.allreduce(fem.assemble_scalar(error_form_p), MPI.SUM))
        error_u = np.sqrt(comm.allreduce(fem.assemble_scalar(error_form_u), MPI.SUM))
        error = np.maximum(error_p, error_u)
        err = error

        uold.x.array[:] = unew.x.array
        pold.x.array[:] = pnew.x.array
        uold.x.scatter_forward()
        pold.x.scatter_forward()

        psi_expr = fem.Expression(psi(unew), V.element.interpolation_points())
        Hold.interpolate(psi_expr)
        Hold.x.scatter_forward()


        if rank == 0:
            print("timestep:", round(t, 5), "in the while loop, iteration is", iter, "error is", round(err, 9), tol)

    if rank == 0:
        print("time step for now is", round(t, 5))
        print("unew max is", np.max(unew.x.array))
        
if rank == 0:
    testend = time.perf_counter()
    print("time consumed:", testend-teststart)