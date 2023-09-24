import numpy as np
import matplotlib.pyplot as plt

# Definição da função que descreve a EDO
def f(t, y):
    return -2 * y

# Parâmetros de simulação
t0 = 0.0  # Tempo inicial
tf = 5.0  # Tempo final
h = 0.1   # Tamanho do passo de tempo
N = int((tf - t0) / h)  # Número de pontos de tempo

# Condição inicial
y0 = 1.0  # Valor inicial de y

# Arrays para armazenar os resultados
time = np.zeros(N+1)
solution = np.zeros(N+1)

# Algoritmo preditor-corretor
time[0] = t0
solution[0] = y0

for i in range(N):
    # Passo de predição (usando Euler)
    y_pred = solution[i] + h * f(time[i], solution[i])

    # Passo de correção (usando RK4)
    k1 = h * f(time[i], solution[i])
    k2 = h * f(time[i] + h/2, solution[i] + k1/2)
    k3 = h * f(time[i] + h/2, solution[i] + k2/2)
    k4 = h * f(time[i] + h, solution[i] + k3)

    y_corr = solution[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    # Atualização das variáveis
    time[i+1] = time[i] + h
    solution[i+1] = y_corr

# Plotagem dos resultados
plt.figure(figsize=(10, 6))
plt.plot(time, solution, label='Solução Numérica')
plt.xlabel('Tempo')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.show()
