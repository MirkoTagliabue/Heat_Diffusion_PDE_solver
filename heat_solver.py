
import numpy as np
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

def phi(t):
    return 0 * t

def psi(t):
    return 0 * t

def h(x):
    return np.sin( np.pi * x )

def f(x,t):
    return 0




# Inizio la funzione principale:

def solve_heat_equation(a, b, T, dx_target, dt_target, phi, psi, h, f):


    # Se dx target non divide b-a, bisogna cambiare dx con un dx più fine (quindi più preciso) che però divida b-a, altrimenti la mesh potrebbe non essere uniforma 
    # per via dell'ultimo intervallo, e questo, lavorando con le differenze finite, complicherebbe molto il codice, dal momento che alcune derivate non si 
    # approssimerebbero considerando lo stesso passo di discretizzazione.
    N_x = int(np.ceil((b - a) / dx_target)) + 1
    mesh_x = np.linspace(a, b, N_x, endpoint=True)
    dx = mesh_x[1] - mesh_x[0]

    # Analogamente, se dt_target non divide T, lo sostituisco con un dt più fine che divida T, col fine di rendere la mesh uniforme:
    N_t = int( np.ceil(T / dt_target) ) + 1
    mesh_t = np.linspace(0, T, N_t, endpoint=True)
    dt = mesh_t[1] - mesh_t[0]


    # Costruisco la matrice A del sistema
    # Dal momento che le uniche incognite sono i nodi interni della mesh_x (i valori dei nodi esterni sono già noti dalle condizioni al contorno), 
    # la matrice A non avrà dimensione N_x X N_x, bensì:  (N_x - 2) X (N_x - 2)
    diag_principale = dt/(dx**2) * 2 * np.ones(N_x - 2) + np.ones(N_x - 2)
    sovra_diag = dt/(dx**2) * (-1) * np.ones(N_x - 3)
    sotto_diag = dt/(dx**2) * (-1) * np.ones(N_x - 3)

    A = np.diag(diag_principale, 0) + np.diag(sovra_diag, 1) + np.diag(sotto_diag, -1)

    # Utilizzo direttamente la fattorizzazione LU su A:
    (LU, piv) = lu_factor(A)


    # Inizializzo la matrice soluzione U di dimensione Nx X Nt 
    # U(:, j) conterrà tutti i valori di temperatura per ogni punto spaziale della mesh_x al tempo t_j
    # U(i, :) conterrà, per il punto x_i \in [a,b] tutti i valori di temperatura per ogni istante di tempo nella mesh_t

    U = np.zeros( (N_x, N_t) )

    # Inizializzo con le condizioni iniziali:
    U[:,0] = h( mesh_x )
    U[0,:] = phi( mesh_t )
    U[-1,:] = psi( mesh_t )

    for m in range(1, N_t):     # $m \in [1, Nt-1]$  poiché uso m per scrivere in U[1:-1 , m]

        # b = U[1:-1, m-1] + f(mesh_x[1:-1], t_{m}) * dt + [phi(t_m), 0, 0, 0, ..., psi(t_m)] * dt / dx^2
        # cioè, NB: evito di usare la prima e l'ultima riga, cioè ragiono solo sui nodi interni

        u_prec = U[1:-1,m-1]    # sarebbe u_m_meno_1

        t_m = mesh_t[m]
        forz_m = f(mesh_x[1:-1], t_m) * dt

        rhs_bordi = np.zeros(N_x-2)     # sarebbe rhs_bordi_m
        rhs_bordi[0] = phi(t_m) * dt / (dx**2)
        rhs_bordi[-1] = psi(t_m) * dt / (dx**2)

        b_m = u_prec + forz_m + rhs_bordi


        # Ora risolvo il sistema Ax=b sfruttando la fattorizzazione LU:
        x = lu_solve( (LU,piv), b_m )


        # Infine scrivo la soluzione ottenuta dentro U
        U[1:-1,m] = x

    #end for m

    return mesh_x, mesh_t, U




######################################################################################################################


# Ora agisco come script e vado a plottare la soluzione.

if __name__ == "__main__":

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



    # Ed infine disegno la soluzione, sfruttando la libreria matplotlib.pyplot

    (X_grid, T_grid) = np.meshgrid(mesh_x, mesh_t, indexing='ij')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X_grid, T_grid, U, cmap='viridis')

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')

    ax.set_title('Soluzione numerica dell\'equazione del calore')

    fig.colorbar(surf)
    plt.show()


