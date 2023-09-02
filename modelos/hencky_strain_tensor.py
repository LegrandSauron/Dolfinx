import ufl
from petsc4py.PETSc import ScalarType
from dolfinx import fem,log, nls
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from dolfinx.io import gmshio

# Scaled variable
carregamento = PETSc.ScalarType(-500)
E = PETSc.ScalarType(78e6)
poisson =PETSc.ScalarType(0.3)
lambda_ =  PETSc.ScalarType(E*poisson / ((1+poisson)*(1-2*poisson)))
G =  PETSc.ScalarType(E / (2 * (1 + poisson)))


domain, cell_tags, facet_tags = gmshio.read_from_msh("malha_estruturada.msh", MPI.COMM_SELF,0, gdim=2)

#definindo o espaço de função:
V = fem.VectorFunctionSpace(domain, ("CG", 1))


"""Definindo as condições de contorno, utilizando dois engastes, lado esquerdo e direito"""

u_bc = np.array((0,) * domain.geometry.dim, dtype=PETSc.ScalarType)
dofs_2 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(10))
bc_1=fem.dirichletbc(u_bc,dofs=dofs_2,V=V)

u_bc = np.array((0,) * domain.geometry.dim, dtype=PETSc.ScalarType)
dofs_2 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(12))
bc_2=fem.dirichletbc(u_bc,dofs=dofs_2,V=V)
bcs= [bc_1,bc_2]


#Definindo os dominios de intergração 
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags)
dx = ufl.Measure("dx", domain=domain)


"""motivos para o uso da função uh = fem.Function(V) , http://home.simula.no/~hpl/homepage/fenics-tutorial/release-1.0-nonabla/webm/nonlinear.html  """
#Aproximação das funções teste e funcao incognita
u  = ufl.TrialFunction(V) # Utilização da função incognita para a forma bilinear 
v  = ufl.TestFunction(V)
uh = fem.Function(V) #Função generica construir


"""Referencia para construção do tensor de cauchy green esquerdo :
 https://jsdokken.com/dolfinx-tutorial/chapter2/hyperelasticity.html#"""

# Spatial dimension
d = len(uh)
# Identity tensor
I = (ufl.Identity(d))
# Deformation gradient
F = (I + ufl.grad(uh))
# Right Cauchy-Green tensor
C = (F * F.T)
# Invariants of deformation tensors
Jhc_e  = ufl.variable(ufl.det(F))
J= Jhc_e


"""Modelo de referencia para a construção do tensor hencky strain : obtenção dos indices do tensor para aplicação do logaritmo sobre os valores nodais :
 https://fenics.readthedocs.io/projects/ufl/en/latest/manual/form_language.html?highlight=as_tensor#defining-indices

Ajuda a respeito da construção do tensor no fenicsx Community
https://fenicsproject.discourse.group/t/logaritm-of-a-tensor/11895 """
#Hencky's Strain
N, M = C.ufl_shape
T_hencky=ufl.as_tensor([[(1/2)*(ufl.ln(C[i,j])) for i in range(N)] for j in range(M)])


#Tensor de tensões 
T_tension= (1/J)*(2.0 * G * T_hencky + lambda_ * ufl.tr(T_hencky) * I )


#Funcao carregamento, carregamento distribuido sobre a face superior atuando para baixo = f(0,y)
f = fem.Constant(domain,ScalarType((0,carregamento)))

#Weak form for  Div T = 0,  Tn= f, u= 0 in 
F_bilinear = ufl.inner(ufl.grad(v),T_tension) * dx - ufl.inner(f, v)*ds

#Definindo o tipo de solucionador.
jacobian= ufl.derivative(F_bilinear,uh ,u)
problem = fem.petsc.NonlinearProblem(F_bilinear,uh,bcs,jacobian)
solver = nls.petsc.NewtonSolver(domain.comm, problem)


# Set Newton solver options
solver.convergence_criterion = "residual"
solver.rtol = 1e-8
solver.report = True
solver.atol = 1e-12

"""COnfigurações para o solver não linear  :
#solver.solver_type = "gmres"
#solver.preconditioner_type = "lu"
#solver.initial_guess = None  # Pode ser um vetor ou None
#solver.divergence_tolerance = 1e-4
#solver.monitor = None  # Pode ser uma função de monitoramento personalizada
#solver.line_search = True
#solver.jacobian_update = "approximate"
#solver.max_it = 51
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

#visualização da convergencia e interações
log.set_log_level(log.LogLevel.INFO)
n, converged = solver.solve(uh)
assert(converged)
print(f"Number of interations: {n:d}")

