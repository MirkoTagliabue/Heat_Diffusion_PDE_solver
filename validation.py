
import numpy as np
import matplotlib.pyplot as plt
from heat_solver import solve_heat_equation

def phi(t):
    return 0 * t

def psi(t):
    return 0 * t

def h(x):
    return np.sin( np.pi * x )

def f(x,t):
    return 0

def u_esatta(x,t):
    return np.exp( -(np.pi**2) * t ) * np.sin( np.pi * x )



# Definisco innanzi tutto i parametri iniziali:

a = 0
b = 1
T = 1

# Idealmente, scegliere qui dx_target in modo tale che dx_target divida b-a 
# e scegliere qui dt_target in modo tale che dt_target divida T
dx_target = 0.01
dt_target = 0.01


# chiamo la funzione che risolve l'equazione del calore, fissati i parametri sopra:
( mesh_x, mesh_t, U ) = solve_heat_equation(a, b, T, dx_target, dt_target, phi, psi, h, f)

N_t = mesh_t.size


# Determino la soluzione esatta:

(X_grid, T_grid) = np.meshgrid(mesh_x, mesh_t, indexing='ij')

U_ex = u_esatta(X_grid, T_grid)




# Ed infine disegno la soluzione, sfruttando la libreria matplotlib.pyplot

fig_1 = plt.figure(figsize=(14, 5),  constrained_layout=True)

ax_1 = fig_1.add_subplot(1, 2, 1, projection='3d')
ax_2 = fig_1.add_subplot(1, 2, 2, projection='3d')

superficie_1 = ax_1.plot_surface(X_grid, T_grid, U_ex, cmap='viridis')
superficie_2 = ax_2.plot_surface(X_grid, T_grid, U, cmap='viridis')

ax_1.set_xlabel('x')
ax_1.set_ylabel('t')
ax_1.set_zlabel('u(x,t)')
ax_1.set_title('Soluzione esatta dell\'equazione del calore')

ax_2.set_xlabel('x')
ax_2.set_ylabel('t')
ax_2.set_zlabel('u(x,t)')
ax_2.set_title('Soluzione approssimata dell\'equazione del calore')


fig_1.colorbar(superficie_1, ax=ax_1, shrink=0.75, pad=0.08)
fig_1.colorbar(superficie_2, ax=ax_2, shrink=0.75, pad=0.08)

plt.show()



# Ora fisso 3 istanti temporali, ad esempio T/3, 2T/3, T ed in tre subplot differenti confronto, per ciascuno di questi 3 istanti temporali,
# la soluzione esatta e quella numerica. Per fare ciò, ricorro a tre parametri modificabili che giocano il ruolo di "percentile in mesh_t"

c_1 = 1/3
c_2 = 2/3
c_3 = 1

m_1 = int( np.round( (N_t - 1) * c_1) )
m_2 = int( np.round( (N_t - 1) * c_2) )
m_3 = int( np.round( (N_t - 1) * c_3) )

t_1 = mesh_t[m_1]
t_2 = mesh_t[m_2]
t_3 = mesh_t[m_3]

Err_1 = np.max( np.abs( U[:,m_1] - U_ex[:,m_1] ) )
Err_2 = np.max( np.abs( U[:,m_2] - U_ex[:,m_2] ) )
Err_3 = np.max( np.abs( U[:,m_3] - U_ex[:,m_3] ) )

fig_2 = plt.figure(figsize=(14, 5),  constrained_layout=True)

ax_1 = fig_2.add_subplot(1, 3, 1)
ax_2 = fig_2.add_subplot(1, 3, 2)
ax_3 = fig_2.add_subplot(1, 3, 3)

ax_1.plot(mesh_x, U_ex[: , m_1] , label='Soluzione esatta')
ax_1.plot(mesh_x, U[: , m_1] , label='Soluzione approssimata')
ax_1.set_xlabel('x')
ax_1.set_ylabel('u(x,t)')
ax_1.set_title(f'Confronto al tempo t_1 = {t_1: 0.2f} \nErrore_1 = {Err_1: 0.3e}')
ax_1.legend()

ax_2.plot(mesh_x, U_ex[: , m_2] , label='Soluzione esatta')
ax_2.plot(mesh_x, U[: , m_2] , label='Soluzione approssimata')
ax_2.set_xlabel('x')
ax_2.set_ylabel('u(x,t)')
ax_2.set_title(f'Confronto al tempo t_2 = {t_2: 0.2f} \nErrore_2 = {Err_2: 0.3e}')
ax_2.legend()

ax_3.plot(mesh_x, U_ex[: , m_3] , label='Soluzione esatta')
ax_3.plot(mesh_x, U[: , m_3] , label='Soluzione approssimata')
ax_3.set_xlabel('x')
ax_3.set_ylabel('u(x,t)')
ax_3.set_title(f'Confronto al tempo t_3 = {t_3: 0.2f} \nErrore_3 = {Err_3: 0.3e}')
ax_3.legend()

plt.show()



# Infine calcolo l'errore complessivo:

Err_tot = np.max( np.abs( U - U_ex ) )

print( f'\tL \' errore di discretizzazione globale vale:\n\tErr_tot = {Err_tot: 0.5e}' )


