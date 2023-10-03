"""
Equações com derivadas temporais de primeira ordem.

Dada a equação diferencial, como exemplo:
                         ∇⋅σ+ρb=ρu¨
Na qual descreve o balanço do momento linear , e sua forma fraca, é dada por :
           ∫Ωρu¨⋅vdx+∫Ωσ(u):ε(v)dx=∫Ωρb⋅vdx+∫∂Ω(σ⋅n)⋅vdsfor all v∈V

Find {u}∈Rn such that {v}T[M]{u¨}+{v}T[K]{u}={v}T{F}for all {v}∈Rn

Find u∈V such that m(u¨,v)+c(u˙,v)+k(u,v)=L(v)for all v∈V


prefixos:

new = n+1 : deslocamento, velocidade referente ao tempo tn+1
n : referente ao deslocamento, velocidade referente ao tempo tn
"""



"""Introduzindo a discretização no tempo"""
#Constantes do metodo de Newmark
Beta= 1
Alpha= 1

# Considerações a respeito do tempo :
T =  500 # Intervalo de tempo total
t_n = 0 # Tempo atual 
t_new = 0 # Tempo posterior 
Steps_tempo = 1000  # 
incremento = T/Steps_tempo

Delta_T =  t_new + t_n  # Intervalo de tempo 

t_zero=0
t_atual= 0


def T_t(T_new,T_old, dt):
    return (T_new - T_old)/dt

def Atualizando_T(T_new, T_old):
    T_old = T_new
    return T_old


for i in range(Steps_tempo):
    t_n += incremento
    print(t_n)
    
    T_t()

    