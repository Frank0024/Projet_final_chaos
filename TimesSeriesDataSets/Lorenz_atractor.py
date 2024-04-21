from PH_method import *


@nb.njit
def lorenz(x, y, z, sigma=10., rho=28., beta=(8./3)):
    """
    Calculate the length of a 2D trajectory using the Lk method for a range of k values.

    Parameters:
    - x, y, z: Arrays of x, y, and z coordinates of the trajectory.
    - height: Height of the plane for finding intercepts.
    - k: Array of k values to use in the Lk method.

    Returns:
    - k: Array of k values used in the Lk method.
    - l_de_k: Array of log of Lk values for each k value.
    """
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    return x_dot, y_dot, z_dot

@nb.njit(parallel=True)
def solve_lorenz(x0=0.0, y0=1.0, z0=1.05, dt=0.01, num_steps=1000000):
    """
    Solve the Lorenz system of ODEs using the Runge-Kutta 4th order method.

    Parameters:
    - x0, y0, z0 (float, optional): Initial conditions for the Lorenz system. Defaults to 0.0, 1.0, and 1.05, respectively.
    - dt (float, optional): Time step. Defaults to 0.01.
    - num_steps (int, optional): Number of time steps. Defaults to 1000000.

    Returns:
    - x, y, z (numpy.ndarray): Arrays of x, y, and z coordinates of the solution of the Lorenz system.
    """
    x, y, z = np.zeros(num_steps + 1), np.zeros(num_steps + 1), np.zeros(num_steps + 1)
    x[0], y[0], z[0] = x0, y0, z0
    for i in nb.prange(num_steps):
        x[i + 1], y[i + 1], z[i + 1] = rk4_step(x[i], y[i], z[i], dt, lorenz)
    return x, y, z

name ="lorenz"

x_graph, y_graph, z_graph = solve_lorenz(num_steps=100000)
plot_3D(name, x_graph, y_graph, z_graph)

x, y, z = solve_lorenz(num_steps=1000000)

best_h = optimise_plane_lk(x, y, z)
x_intercept_best, y_intercept_best = find_intercepts(x, y, z, best_h) 
k_best, l_of_k_best = find_lk_for_k(x, y, z, best_h)
plot_intercept(name, x_intercept_best, y_intercept_best, best_h)
plot_LK(name, k_best, l_of_k_best, best_h)

worst_h = optimise_plane_lk(x, y, z, max_diff=False)
x_intercept_worst, y_intercept_worst = find_intercepts(x, y, z, worst_h) 
k_worst, l_of_k_worst = find_lk_for_k(x, y, z, worst_h)
plot_intercept(name, x_intercept_worst, y_intercept_worst, worst_h, best=False)
plot_LK(name, k_worst, l_of_k_worst, worst_h, best=False)
