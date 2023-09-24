
# Fenics-related packages

from dolfinx import fem, mesh,log,nls
import numpy as np
import math
# Plotting packages
import matplotlib.pyplot as plt
# Current time package
from datetime import datetime
from sympy import dirichlet_eta
import ufl


# Set level of detail for log messages (integer)
# 
# Guide:
# CRITICAL  = 50, // errors that may lead to data corruption
# ERROR     = 40, // things that HAVE gone wrong
# WARNING   = 30, // things that MAY go wrong later
# INFO      = 20, // information of general interest (includes solver info)
# PROGRESS  = 16, // what's happening (broadly)
# TRACE     = 13, // what's happening (in detail)
# DBG       = 10  // sundry
#log.set_log_level(30)


#-----------------------------------------------------------
# Global Fenics parameters
# parameters["form_compiler"]["cpp_optimize"]=True
# parameters["form_compiler"]["optimize"] = True
# set_log_level(30)

# The behavior of the form compiler FFC can be adjusted by prescribing
# various parameters. Here, we want to use the UFLACS backend of FFC::

# Optimization options for the form compiler
#parameters["form_compiler"]["cpp_optimize"] = True
#parameters["form_compiler"]["representation"] = "uflacs"
#parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
#quadDegree = 2
#parameters["form_compiler"]["quadrature_degree"] = quadDegree


from dolfinx import fem, mesh,log,nls
import numpy as np
import math
# Plotting packages
import matplotlib.pyplot as plt
# Current time package
from datetime import datetime
from sympy import dirichlet_eta
import ufl
from mpi4py import MPI

# Dimensions
scaleX = 1.0e4
xElem = 1

scaleY = 1400.e0 

scaleZ = 1.0e4
zElem = 1

# N number of elements in y-direction                                             
N = 128

#scaleZ = 30.e0
int1 = 200.e0
int2 = scaleY/2.-int1
int3 = int1/2.0 + int2
int4 = 500.e0
                
M1 = 60. #N/scaleY*int1
M2 = 2.0 #90 #int2/scaleY*N
M3 = M1/2
r1 = 1/1.5
r2 = 1/1.06     
r3 = r1
a1 = (1-r1)/(1-r1**M1)
a2 = (1-r2)/(1-r2**(M2)) 
a3 = (1-r3)/(1-r3**M3)
                
preMapLength = float(int1 + 2*M2*(int1/M1))

domain =mesh.create_box(MPI.COMM_WORLD,[(0.,-preMapLength, 0.0),(scaleX, preMapLength, scaleZ)],[xElem,N, zElem])

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "resultados/dolfinx_version.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)


