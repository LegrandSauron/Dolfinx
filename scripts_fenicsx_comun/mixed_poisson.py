"""Ok"""

import numpy as np
import pyvista

from ufl import FiniteElement, MixedElement, VectorElement
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import(Measure, SpatialCoordinate, FacetNormal, TestFunctions, TrialFunctions,
                div, grad, exp, inner, sin)

from mpi4py import MPI
from petsc4py import PETSc

"""https://fenicsproject.discourse.group/t/reformulating-mixed-poisson-problem/12430"""
domain = mesh.create_unit_square(MPI.COMM_WORLD, 12, 12, mesh.CellType.triangle)

k = 1
Q_el = VectorElement("Lagrange", domain.ufl_cell(), k)
P_el = FiniteElement("Lagrange", domain.ufl_cell(), k)
V_el = MixedElement([Q_el, P_el])
V = fem.FunctionSpace(domain, V_el)

(sigma, u) = TrialFunctions(V)
(tau, v) = TestFunctions(V)

x = SpatialCoordinate(domain)
f = 10.0 * exp(-((x[0] - 0.5)**2 + (x[1] - 0.5)**2) / 0.02)
g = sin(5 * x[0])

dx = Measure("dx", domain)

a = (inner(sigma, tau) + u * div(tau) - inner(sigma, grad(v))) * dx
L = - f * v * dx

boundaries = [(1, lambda x: np.isclose(x[1], 0.0)),
              (2, lambda x: np.isclose(x[0], 1.0)),
              (3, lambda x: np.isclose(x[1], 1.0)),
              (4, lambda x: np.isclose(x[0], 0.0))]

facet_indices, facet_markers = [], []
fdim = domain.topology.dim - 1
for (marker, locator) in boundaries:
    facets = mesh.locate_entities(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

facet_top = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
Q, _ = V.sub(0).collapse()
dofs_top = fem.locate_dofs_topological((V.sub(0), Q), fdim, facet_top)

ds = Measure("ds", domain=domain, subdomain_data=facet_tag)

n = FacetNormal(domain)

a += - u * inner(tau, n) * ds(1) - u * inner(tau, n) * ds(3)\
    + v * inner(sigma, n) * ds(2) + v * inner(sigma, n) * ds(4)
L += - g * v * ds(1) - g * v * ds(3)

problem = LinearProblem(a, L, petsc_options={"ksp_type": "preonly",
                                                  "pc_type": "lu",
                                                  "pc_factor_mat_solver_type": "mumps"})

w_h = problem.solve()

sigma_h, u_h = w_h.sub(0).collapse(), w_h.sub(1).collapse()

P, _ = V.sub(1).collapse()
grid = pyvista.UnstructuredGrid(*plot.create_vtk_mesh(P))
grid.point_data["u"] = u_h.x.array.real
grid.set_active_scalars("u")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
plotter.view_xy()
plotter.show()