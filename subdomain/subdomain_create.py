



from mpi4py import MPI
from dolfinx.io import gmshio


domain, ct, facet_tags = gmshio.read_from_msh("malha_com_eletrodo_3d.msh", MPI.COMM_WORLD,0, gdim=3)

"""
Obtenção das faces para aplicação das condições de contorno são dadas por :

1 14 "face_superior_eletrodo_cima"
1 15 "face_inferior_eletrodo_cima"
1 16 "face_inferior_eletrodo_baixo"
1 17 "face_superior_eletrodo_baixo"
1 21 "gel_polimerico_lado_esquerdo"
1 22 "gel_polimerico_lado_direito"
1 23 "eletrodo_cima_lado_direito"
1 24 "eletrodo_baixo_lado_direito"
1 25 "eletrodo_baixo_lado_esquerda"
1 26 "eletrodo_cima_lado_esquerda"
2 18 "eletrodo_baixo"
2 19 "eletrodo_cima"
2 20 "gel_polimerico"
2 27 "dominio_eletrodo_gel" 

"""
#Q = fem.FunctionSpace(domain, ("DG", 0))

#Emod = fem.Function(Q)
#concrete = ct.find(18)
#Emod.x.array[concrete] = np.full_like(concrete, 4.3e6, dtype=PETSc.ScalarType)
#steel = ct.find(19)
#Emod.x.array[steel]  = np.full_like(steel, 30e6, dtype=PETSc.ScalarType)

