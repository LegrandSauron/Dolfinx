"""Aplicação de uma condição de contorno dependente do tempo """
""'https://fenicsproject.discourse.group/t/dirichletbc-depending-on-both-time-and-another-function/12475"""

x = dolfinx.fem.Function(V)
t = dolfinx.fem.Constant(mesh, np.float64(0.0))
T = dolfinx.fem.Constant(mesh, np.float64(10))
u_d = (1 + t * T) * t + 2 * T * x
ud_expr = dolfinx.fem.Expression(u_d, V.element.interpolation_points())
u_D = dolfinx.fem.Function(V)
u_D.interpolate(ud_expr)
t.value += dt
u_D.interpolate(ud_expr)