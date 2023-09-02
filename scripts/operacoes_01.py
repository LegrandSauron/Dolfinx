
carregamento= 50000
E = 78e6
poisson = 0.3
lambda_ = E*poisson / ((1+poisson)*(1-2*poisson))
G= E / 2*(1+poisson)


"""Criando um tensor de 2° e 1° ordem e realizando o produto escalar entre eles"""
E=ufl.as_tensor([[(1/2)*(ufl.ln(C[i,j])) for i in range(N)] for j in range(M)])
T_tensao= 2.0 * G * E + lambda_ * ufl.tr(E) * I 


# Verificando o rank (ordem) do tensor no espaço vetorial
#print("Rank do tensor no espaço vetorial:", ufl.rank(E))


#verificando a ordem dos tensores no espaço vetorial e de funçoes 
"""Espaço vetorial"""
#print(ufl.rank(u)) 
#print(ufl.rank(ufl.grad(u)))
#print(ufl.rank(ufl.div(u)))


"""Espaço de funções"""
#print(ufl.rank(uh))
#print(ufl.rank(ufl.grad(uh)))
#print(ufl.rank(ufl.div(uh)))


"""Espaço de tensores"""
#print(ufl.rank(ut))
#print(ufl.rank(ufl.grad(ut)))
#print(ufl.rank(ufl.div(ut)))
Fa =(I + ufl.grad(u)) 
print(ufl.rank(Fa))