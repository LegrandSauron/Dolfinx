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
n : reference ao deslocamento, velocidade referente ao tempo tn
"""



"""Introduzindo a discretização no tempo"""
#Constantes do metodo de Newmark
Beta= 1
Alpha= 1

# Considerações a respeito do tempo :
T = [0, 500] # Intervalo de tempo total
t_n = 0 # Tempo atual 
t_new = 0 # Tempo posterior 
Steps_tempo = 1000  # 
incremento = T/Steps_tempo

Delta_T =  t_new + t_n  # Intervalo de tempo 



"""Solução em u_new""" 

u_n = "Deslocamento "
v_n = "Velocidade"
a_n = "Aceleração"



def u_new(u_n, v_n, a_n,a_new):
    return u_n + Delta_T*v_n + ((Delta_T**2)/2)*((1- 2*Beta)*a_n + 2*Beta*a_new)

Velocity_new =  32
