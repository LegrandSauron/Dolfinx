
"""
#Formulação variacional para Div T = 0, em Tn= f,u= 0 in facet_tags.find(16)
problem = fem.petsc.NonlinearProblem(F=F_bilinear ,u=u,bcs=[bc])


#solucao não linear 
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "residual"
solver.rtol = 1e-8
solver.max_it = 30
solver.report = True

solver.atol = 1e-12
solver.preconditioner_type = "gmres"
solver.initial_guess = None  # Pode ser um vetor ou None
solver.divergence_tolerance = 1e-6
solver.monitor = None  # Pode ser uma função de monitoramento personalizada
solver.line_search = True
solver.jacobian_update = "approximate"
solver.error_on_nonconvergence = True

"""
"""ao linear
ksp = solver.krylov_solver
opts = PETSc.Options()

option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()
"""
from dolfinx import fem
import ufl
from petsc4py.PETSc import ScalarType
from dolfinx import fem,log, nls ,mesh, plot
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc


# Scaled variable
carregamento= -5000
E = 78e6
poisson = 0.3
lambda_ = E*poisson / ((1+poisson)*(1-2*poisson))
G= E / 2*(1+poisson)




domain, cell_tags, facet_tags = gmshio.read_from_msh("malha_150_60_acopla.msh", MPI.COMM_WORLD,0, gdim=2)

V = fem.VectorFunctionSpace(domain, ("CG", 1))

#condições de contorno
u_D = np.array([0,0], dtype=ScalarType) 
dofs_2 = fem.locate_dofs_topological(V, facet_tags.dim, facet_tags.find(16))
bc=fem.dirichletbc(u_D,dofs=dofs_2,V=V)

#ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags)


#application of boundary conditions , crimping.
""" Faces para aplicação das condições de contorno
Physical Curve("esgaste", 16) = {14, 15, 3, 1};
Physical Curve("carregamento", 17) = {8};
Physical Surface("dominio", 18) = {1, 2, 3, 4};
"""

#Aproximação das funções teste e funcao incognita
v = ufl.TestFunction(V)
u= fem.Function(V)
#u= ufl.TrialFunction(V)
#hencky's strain tensor
d = len(u)
I = (ufl.Identity(d)) 
F =(I + ufl.grad(u)) 

#tensor de cauchy-Green left 
C= (F * F.T )


#determinando o Jacobiano
Jhc_e= ufl.det(F)
J= Jhc_e

#Hencky's Strain
N, M = C.ufl_shape
#T_hencky=ufl.as_tensor([[(1/2)*(ufl.ln(C[i,j])) for i in range(N)] for j in range(M)])

def T_hencky(C):
    return ufl.as_tensor([[(1/2)*(ufl.ln(C[i,j])) for i in range(N)] for j in range(M)])


#Tensor de tensões
#T_tension= (1/J)*(2.0 * G * T_hencky + lambda_ * ufl.tr(T_hencky) * I )

def T_tension():
    return(1/J)*(2.0 * G * T_hencky(C) + lambda_ * ufl.tr(T_hencky(C)) * I )


#Funcao carregamento = f(0,y)
f = fem.Constant(domain, ScalarType((0,carregamento )))


#Formulação variacional para Div T = 0, em Tn= f,u= 0 em x=0
#a = ufl.inner(ufl.grad(v),T_tension) * ufl.dx
#L = ufl.inner(f, v) * ufl.ds


#Weak form for  Div T = 0,  Tn= f, u= 0 in 
F_bilinear =fem.form(ufl.inner(ufl.grad(v),T_tension) * ufl.dx )
L= ufl.inner(f, v) *ufl.ds

#Criando um solver do  não linear


A = fem.petsc.assemble_matrix(F_bilinear, bcs=[bc])
#A.assemble()

#b = fem.petsc.assemble_vector(L)
from scripts.solver_manual import*

print(T_hencky)

solver = PETSc.SNES().create(domain.comm)
solver.setFromOptions()
#solver.setOperators(A)
