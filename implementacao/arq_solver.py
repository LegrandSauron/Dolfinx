from dolfinx import fem
import dolfinx
import ufl
from ufl import ds, dx, grad, inner, dot
from petsc4py.PETSc import ScalarType
from dolfinx.io import XDMFFile
from dolfinx import fem, mesh, log, nls
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc 
from petsc4py.PETSc import*
import petsc4py

# Scaled variable
carregamento= 50000
E = 78e6
poisson = 0.3
lambda_ = E*poisson / ((1+poisson)*(1-2*poisson))
G= E / 2*(1+poisson)



#Criando ou importando a geometria e realizando a discretização.
domain= mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([2,1])],
                  [200,200], cell_type=mesh.CellType.triangle)

#Importação da geometria e das condições de contorno.
#malha, cell_tags, facet_tags = gmshio.read_from_msh("malha_ece.msh", MPI.COMM_SELF,0, gdim=2)


#Realizando a aproximação do dominio por um espaço de funcao V  
V = fem.VectorFunctionSpace(domain, ("CG", 1))


#Definindo as condições de contorno
def clamped_boundary(x):
    return np.isclose(x[0], 0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
u_D = np.array([0,0], dtype=ScalarType)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
T = fem.Constant(domain, ScalarType((0, 0)))

#Espaço de integração no contorno da geometria 
#ds = ufl.Measure("ds", domain=domain, subdomain_data= boundary_facets)
#dx = ufl.Measure("dx", domain=domain)


#Aproximação das funções teste e funcao incognita
v = ufl.TestFunction(V)
uh = fem.Function(V)
#u = ufl.TrialFunction(V)


# Spatial dimension
d = len(uh)

# Identity tensor
I = ufl.variable(ufl.Identity(d))

# Deformation gradient
F = ufl.variable(I + ufl.grad(uh))

# Right Cauchy-Green tensor
C = ufl.variable(F * F.T)


def Hencky_Strain(Cauchy_G_E):
    N,M = Cauchy_G_E.ufl_shape
    return ufl.as_tensor([[0.5*(ufl.ln(Cauchy_G_E[i,j])) for i in range(N)] for j in range(M)])

J  = ufl.variable(ufl.det(F))

#Definindo o tensor de tensões T
T_tensao = (1/J)*(2.0 * G * Hencky_Strain(C) + lambda_ * ufl.tr(Hencky_Strain(C)) *I )

#funcao carregamento
f = fem.Constant(domain, ScalarType((0,carregamento )))

#formulação fraca a partir do funcional bilinear

a =ufl.inner(ufl.grad(v),T_tensao) * ufl.dx - ufl.dot(f, v)*ufl.ds 
#L = - ufl.inner(f, v)*ufl.ds 

problem = fem.petsc.NonlinearProblem(F=a, u=uh, bcs=[bc])



#bilinear_form = fem.form(a)

#b = fem.petsc.create_vector(bilinear_form)
#A = fem.petsc.assemble_matrix(bilinear_form, bcs=[bc])





"""
Metodo de solucao de um sistema não linear: utiza-se o metodo de newton para solucão, inicialmente temos que o sistema deve ser lineralizado , AX= B, para isso, obtem-se o sistema linear atraves da matriz jacobiana J*S= F
"""


from dolfinx import nls
solver = nls.petsc.NewtonSolver(domain.comm, problem)


#Set Newton solver options
solver.atol = 1e-12
solver.rtol = 1e-12
solver.convergence_criterion = "residual"
solver.report = True
   

#config
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_max_it"] = 1000
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"

ksp.setFromOptions()



log.set_log_level(log.LogLevel.INFO)
n, converged = solver.solve(u=uh)
assert(converged)
print(f"Number of interations: {n:d}")

