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
u(a,t)=\phi(t), \qquad u(b,t)=\psi(t), \qquad \forall t\in[0,T],
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
a = x_0 < x_1 < x_2 < ... < x_{N_x-1} < x_{N_x} = b
$$

and

$$
0 = t_0 < t_1 < t_2 < ... < t_{N_t-1} < t_{N_t} = T
$$

The sets of those points are called *meshes*, so:

$$ 
mesh \textunderscore x = \\{ a, x_1, x_2, ..., x_{N_x} \\},   \qquad 
mesh \textunderscore t = \\{ 0, t_1, t_2, ..., t_{N_t} \\}
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

[work in progress]

