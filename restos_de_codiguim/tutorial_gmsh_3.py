
import gmsh
from dolfinx.io import XDMFFile, gmshio

from mpi4py import MPI
# Inicializar o Gmsh
gmsh.initialize()

# Criar um novo modelo vazio
gmsh.model.add('Volume_exemplo')

# Definindo os pontos
ponto1 = gmsh.model.geo.addPoint(0, 0, 0)
ponto2 = gmsh.model.geo.addPoint(1, 0, 0)
ponto3 = gmsh.model.geo.addPoint(0, 2, 0)
ponto4 = gmsh.model.geo.addPoint(1, 2, 0)

# Criar linhas entre dois pontos
linha1 = gmsh.model.geo.addLine(ponto1, ponto2)
linha2 = gmsh.model.geo.addLine(ponto2, ponto3)
linha3 = gmsh.model.geo.addLine(ponto3, ponto4)
linha4 = gmsh.model.geo.addLine(ponto4, ponto1)
    
#Definir um plano a partir das linhas criadas

plano= gmsh.model.geo.add_curve_loop([linha1,linha2,linha3,linha4])

#Definindo uma superficiel 2d
plano_2d = gmsh.model.geo.addPlaneSurface([plano])

# Definir uma propriedade física para a superfície (por exemplo, uma condição de contorno)
Entidade_1=gmsh.model.getEntities(dim=2)

faces = gmsh.model.occ.getEntities(dim=1)

# Definir a visibilidade das entidades criadas
gmsh.model.addPhysicalGroup(2, [plano_2d], tag=1)
gmsh.model.setPhysicalName(1, 1, "Plano")
gmsh.model.occ.synchronize()

# Gerar a malha
gmsh.model.mesh.generate(2)

gmsh.model.occ.synchronize()
# Salvar o modelo em um arquivo .msh
gmsh.write("modelo.msh")

gmsh.fltk.run()
# Fechar o Gmsh

#mesh = gmshio.model_to_mesh(gmsh.model,MPI.COMM_SELF,0,2)