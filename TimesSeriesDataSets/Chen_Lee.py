from PH_method import *

@nb.njit
def chen_lee(x, y, z, a=5.0, b=-10.0, c=-0.38):
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
    x_dot = (a * x) - (y * z)
    y_dot = (x * z) + (b * y)
    z_dot = ((x * y) / 3) + (c * z)
    return x_dot, y_dot, z_dot

@nb.njit(parallel=True)
def solve_chen_lee(x0=1.0, y0=1.0, z0=1.0, dt=0.01, num_steps=1000000):
    """
    Solve the Lorenz system of ODEs using the Runge-Kutta 4th order method.
    """
    x, y, z = np.zeros(num_steps + 1), np.zeros(num_steps + 1), np.zeros(num_steps + 1)
    x[0], y[0], z[0] = x0, y0, z0
    for i in nb.prange(num_steps):
        x[i + 1], y[i + 1], z[i + 1] = rk4_step(x[i], y[i], z[i], dt, chen_lee)
    return x, y, z

name ="chenlee"

#x_graph, y_graph, z_graph = solve_chen_lee(num_steps=100000)
#plot_3D(name, x_graph, y_graph, z_graph, cmap=plt.cm.plasma)

x, y, z = solve_chen_lee(num_steps=1000000)

best_h = optimise_plane_lk(x, y, z, height_start=5, height_stop=14, step=0.1)
x_intercept_best, y_intercept_best = find_intercepts(x, y, z, best_h) 
k_best, l_of_k_best = find_lk_for_k(x, y, z, best_h)
plot_intercept(name, x_intercept_best, y_intercept_best, best_h)
plot_LK(name, k_best, l_of_k_best, best_h)

worst_h = optimise_plane_lk(x, y, z, height_start=5, height_stop=14, step=0.1, max_diff=False)
x_intercept_worst, y_intercept_worst = find_intercepts(x, y, z, worst_h) 
k_worst, l_of_k_worst = find_lk_for_k(x, y, z, worst_h)
plot_intercept(name, x_intercept_worst, y_intercept_worst, worst_h, best=False)
plot_LK(name, k_worst, l_of_k_worst, worst_h, best=False)