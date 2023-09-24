import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do sistema
m = 1.0    # Massa
k = 10.0   # Constante da mola
c = 0.5    # Coeficiente de amortecimento
F0 = 1.0   # Amplitude da força externa
omega = 2.0  # Frequência da força externa

# Parâmetros de simulação
dt = 0.01  # Passo de tempo
T = 10.0   # Tempo total
N = int(T / dt)  # Número de pontos de tempo

# Parâmetros do método de Newmark
beta = 0.25
gamma = 0.5

# Condições iniciais
u0 = 0.0  # Posição inicial
v0 = 0.0  # Velocidade inicial

# Arrays para armazenar resultados
time = np.zeros(N)
displacement = np.zeros(N)
velocity = np.zeros(N)

# Implementação do método de Newmark
u = u0
v = v0
for i in range(N):
    time[i] = i * dt
    displacement[i] = u

    # Força externa no tempo atual
    F = F0 * np.cos(omega * time[i])

    # Predição da aceleração
    a_pred = (F - c * v - k * u) / m

    # Atualização das posições e velocidades
    u_new = u + dt * v + (1 - 2 * beta) * (dt ** 2) * a_pred / 2
    v_new = v + dt * ((1 - gamma) * a_pred + gamma * a_pred)

    # Atualização das variáveis
    u = u_new
    v = v_new

# Plotagem dos resultados
plt.figure(figsize=(10, 6))
plt.plot(time, displacement, label='Deslocamento')
plt.xlabel('Tempo')
plt.ylabel('Deslocamento')
plt.legend()
plt.grid(True)
plt.show()
