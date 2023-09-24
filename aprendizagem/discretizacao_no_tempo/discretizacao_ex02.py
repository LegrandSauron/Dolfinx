import numpy as np
import matplotlib.pyplot as plt

# Definição do domínio
L = 1.0  # Comprimento do domínio
num_elements = 10  # Número de elementos finitos
element_length = L / num_elements

# Nós e elementos
nodes = np.linspace(0, L, num_elements + 1)
elements = [(i, i + 1) for i in range(num_elements)]

# Funções de base (funções de interpolação)
def basis_function(x, node1, node2):
    return (node2 - x) / (node2 - node1), (x - node1) / (node2 - node1)

# Montagem da matriz de rigidez global e do vetor de carga global
K = np.zeros((num_elements + 1, num_elements + 1))
F = np.zeros(num_elements + 1)

for element in elements:
    node1, node2 = element
    for i in range(2):
        for j in range(2):
            integral = np.trapz([basis_function(node1, node1, node2)[i] * basis_function(x, node1, node2)[j]
                                  for x in np.linspace(node1, node2, 100)], dx=element_length)
            K[node1, node1] += integral
            K[node2, node2] += integral
            K[node1, node2] -= integral
            K[node2, node1] -= integral

# Resolução do sistema KU = F
U = np.linalg.solve(K, F)

# Plotagem da solução
plt.figure(figsize=(10, 6))
plt.plot(nodes, U, marker='o', linestyle='-', label='Solução Numérica')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid(True)
plt.show()
