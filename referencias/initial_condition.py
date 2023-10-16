def y0_init(x):
    values = numpy.zeros((1, x.shape[1]))
    values[0] = 1.0
    return values

def y1_init(x):
    values = numpy.zeros((1, x.shape[1]))
    values[0] = 2.0
    return values


msh = mesh.create_interval(MPI.COMM_WORLD, 10, points=(0, 1))

x = SpatialCoordinate(msh)

CG1_elem = FiniteElement("CG", msh.ufl_cell(), 1)
ME_elem = MixedElement([CG1_elem, CG1_elem])
ME = FunctionSpace(msh, ME_elem)

y = Function(ME)
y.sub(0).interpolate(y0_init)
y.sub(1).interpolate(y1_init)