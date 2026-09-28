

# 1D Heat Diffusion PDE Solver

This project implements an implicit finite-difference method in Python for the numerical solution of the one-dimensional heat diffusion 
partial differential equation with a source term and Dirichlet boundary conditions.  
The solver is implemented using the libraries NumPy and SciPy, while Matplotlib is used to visualize the numerical solution and compare it with an 
analytical solution in a test case where (for a particular choice of boundary and initial conditions) the exact solution is known in closed form.

---

## Problem formulation

We consider the following partial differential equation:

$$
\frac{\partial u}{\partial t}(x,t) - \frac{\partial^2 u}{\partial x^2}(x,t) = f(x,t),
\qquad
\forall x\in[a,b],\quad \forall t\in[0,T].
$$

The equation is supplemented with Dirichlet boundary conditions of the form:

$$
u(a,t)=\varphi(t), \qquad u(b,t)=\psi(t), \qquad \forall t\in[0,T],
$$

and the initial condition:

$$
u(x,0)=h(x),  \qquad  \forall x\in[a,b].
$$

The goal is to approximate the solution $u(x,t)$ on the whole space-time domain $[a,b] \times [0,T]$.  

The equation $u_t - u_{xx} = f$ is called a *diffusion equation* and is a partial differential equation used to model for example the diffusion of heat 
on a bar or the diffusion of a colorant in a river. 

---

## Space-time discretization

The spatial interval $[a,b]$ and the time interval $[0,T]$ are partitioned into sub-intervals whose endpoints are:

$$
a = x_1 < x_2 < x_3 < ... < x_{N_x-1} < x_{N_x} = b
$$

and

$$
0 = t_1 < t_2 < t_3 < ... < t_{N_t-1} < t_{N_t} = T
$$

The sets of those points are called *meshes*, so:

$$ 
mesh \textunderscore x = \\{ a, x_2, x_3, ..., x_{N_x} \\},   \qquad 
mesh \textunderscore t = \\{ 0, t_2, t_3, ..., t_{N_t} \\}
$$

The elements of the meshes are called *nodes*. If the distance of two contiguous points is always constant the mesh is said to be *uniform*.
In our implementation we use uniform meshes, and so, we can call $\Delta x$ the distance between two contiguous nodes in the spatial mesh $mesh \textunderscore x$ and 
we can call $\Delta t$ the distance of two contiguous points in the temporal mesh $mesh \textunderscore t$.

In what follows, the value $u(x_j,t_m)$ will be denoted by

$$
u_j^m = u(x_j,t_m).
$$




### Finite-difference operators

For a sufficiently regular function $v(x)$, the forward and backward finite differences are defined as

$$
D_{+} \left[ v(x_j) \right] = \frac{ v(x_{j+1}) - v(x_j) }{\Delta x},
$$

and

$$
D_{-} \left[ v(x_j) \right] = \frac{ v(x_j) - v(x_{j-1}) }{\Delta x}.
$$

Both operators approximate the first derivative at the point $x_j$ with first-order accuracy:

$$
D_{+} v(x_j) = v'(x_j) + O(\Delta x),
$$

$$
D_{-} v(x_j) = v'(x_j) + O(\Delta x).
$$

The centered second-order finite difference is defined as

$$
D^2 \left[ v(x_j) \right] = \frac{ v(x_{j+1}) - 2v(x_j) + v(x_{j-1}) }{\Delta x^2},
$$

and approximates the second derivative with second-order accuracy:

$$
D^2 \left[ v(x_j) \right] = v''(x_j) + O(\Delta x^2).
$$

Moreover, the centered second difference can be obtained by composing the backward and forward differences:

$$
D^2v(x_j) = D_{-} \left[ D_{+} v(x_j) \right]  = \frac{ D_{+} \left[ v(x_j) \right] - D_{+} \left[ v(x_{j-1}) \right] }{ \Delta x } 
          = \frac{ v(x_{j+1}) - 2v(x_j) + v(x_{j-1}) }{\Delta x^2}.
$$

For the time derivative, a backward finite difference is used:

$$
\frac{\partial u}{\partial t}(x_j,t_m)
\approx
\frac{u_j^m - u_j^{m-1}}{\Delta t},
$$

which is first-order accurate in time:

$$
\frac{u_j^m - u_j^{m-1}}{\Delta t} = 
u_t(x_j,t_m) + O(\Delta t).
$$

---

## Implicit finite-difference scheme

Using the backward finite difference for the time derivative and the centered finite difference for the second spatial derivative, the numerical method
for approximating the given heat equation becomes:


$$
\frac{ u_j^m - u_j^{m-1} }{\Delta t} - \frac{ u_{j+1}^m - 2u_j^m + u_{j-1}^m }{ \Delta x^2 }  =  f(x_j,t_m).
$$

Multiplying by $\Delta t$ and rearranging the terms, we obtain:

$$
u_j^m - \left( u_{j+1}^m - 2u_j^m + u_{j-1}^m \right) \cdot \frac{ \Delta t}{ \Delta x^2 } =  f(x_j,t_m) \cdot \Delta t  +  u_j^{m-1}.
$$

Since the values at the boundary nodes are already known from the Dirichlet conditions, only the internal nodes $x_j$ for $j=2,\ldots, N_x-1$ are unknown, in fact:

$$
\forall m = 1, ..., N_t  \qquad u(x_1, t_m) = u(a, t_m) = \varphi(t_m) ,  \qquad  u(x_{N_x}, t_m) = u(b, t_m) = \psi(t_m)
$$

At every time step $t_m$, the method therefore requires the solution of a linear system

$$
A \cdot x_m = b_m 
$$

where $A$ is the $(N_x - 2) \times (N_x - 2)$ matrix of the form:

$$
A=
\begin{pmatrix}
1+2\frac{\Delta t}{\Delta x^2}   &    -\frac{\Delta t}{\Delta x^2}      &                                  &                                         \\
-\frac{\Delta t}{\Delta x^2}     &    1+2\frac{\Delta t}{\Delta x^2}    & -\frac{\Delta t}{\Delta x^2}     &                                         \\
                                 &    \ddots                            & \ddots                           & \ddots                                  \\
                                 &                                      & -\frac{\Delta t}{\Delta x^2}     & 1+2\frac{\Delta t}{\Delta x^2}
\end{pmatrix}.
$$

The right-hand side $b_m$ can be written schematically as

$$
b_m \hspace{0.2cm} = \hspace{0.2cm}  u_{\mathrm{prec}} \hspace{0.2cm} + \hspace{0.2cm} 
\Delta t \cdot f^m \hspace{0.2cm} + \hspace{0.2cm} 
\mathrm{RHS}_{\mathrm{bordo}} \hspace{0.1cm}  ,
$$

where:

$$
u_{\mathrm{prec}} = 
\begin{pmatrix}
u_2^{m-1}  \\
u_3^{m-1}  \\
\vdots     \\
u_{N_x-1}^{m-1}
\end{pmatrix},
                    \hspace{2cm}
f^m = 
\begin{pmatrix}
f(x_2,t_m)  \\
f(x_3,t_m)  \\
\vdots     \\
f(x_{N_x-1},t_m)
\end{pmatrix},
                    \hspace{2cm}
\mathrm{RHS}_{\mathrm{bordo}} \hspace{0.1cm} = \hspace{0.1cm}  
\frac{\Delta t}{\Delta x^2}  \cdot 
\begin{pmatrix}
\varphi(t_m) \\
0         \\
\vdots    \\
0         \\
\psi(t_m)
\end{pmatrix}.                    
$$

The matrix $A$ does not depend on the time index $m$, therefore it is possible to compute its $LU$ factorization once in the beginning 
and reuse it at every time step in order to optimize the computational cost of the resolution of the given linear systems.  

Since the numerical method requires the resolution of systems of equations at every step, it is implicit. This numerical scheme is known as *Backward Euler*, in 
contrast to the explicit method *Forward Euler*. The latter does not require the solution of a linear system at each time step, but as an explicit method, 
its region of stability imposes a restriction on the relation between $\Delta x$ and $\Delta t$.

The resulting method is first-order accurate in time and second-order accurate in space, this means that:

$$
\mathrm{Err} = O(\Delta t + \Delta x^2).
$$

Moreover, as the initial condition imposes:

$$
u(x_j,0) = h(x_j)  \qquad \forall j = 1,2, \dots, N_x,
$$

for this reason, for $m=1$ it is not required to solve a linear system. In fact, the assignment of the values $h(x_j)$ to $u(x_j,0)$ acts as the initial base case 
that allows the method to start.

---

## Validation against an analytical solution

The numerical method has been tested on a problem for which the exact solution is known.

