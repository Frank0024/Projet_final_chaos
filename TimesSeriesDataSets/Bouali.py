from PH_method import *

@nb.njit
def bouali(x, y, z, a=4.0, b=1.0, c=1.5, s=1.0, alpha=0.3, beta=0.05):
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
    x_dot = x * (a - y) + (alpha * z)
    y_dot = -y * (b - x**2)
    z_dot = -x * (c - (s * z)) - (beta * z)
    return x_dot, y_dot, z_dot

@nb.njit(parallel=True)
def solve_bouali(x0=0.5, y0=0.5, z0=0.5, dt=0.01, num_steps=1000000):
    """
    Solve the Lorenz system of ODEs using the Runge-Kutta 4th order method.
    """
    x, y, z = np.zeros(num_steps + 1), np.zeros(num_steps + 1), np.zeros(num_steps + 1)
    x[0], y[0], z[0] = x0, y0, z0
    for i in nb.prange(num_steps):
        x[i + 1], y[i + 1], z[i + 1] = rk4_step(x[i], y[i], z[i], dt, bouali)
    return x, y, z

name ="bouali"

#x_graph, y_graph, z_graph = solve_bouali(num_steps=100000)
#plot_3D(name, x_graph, y_graph, z_graph, cmap=plt.cm.cividis)

x, y, z = solve_bouali(num_steps=1000000)

best_h = optimise_plane_lk(x, y, z, height_start=-6, height_stop=1, step=0.1)
x_intercept_best, y_intercept_best = find_intercepts(x, y, z, best_h) 
k_best, l_of_k_best = find_lk_for_k(x, y, z, best_h)
plot_intercept(name, x_intercept_best, y_intercept_best, best_h)
plot_LK(name, k_best, l_of_k_best, best_h)

worst_h = optimise_plane_lk(x, y, z, height_start=-6,height_stop=1,step=0.1, max_diff=False)
x_intercept_worst, y_intercept_worst = find_intercepts(x, y, z, worst_h) 
k_worst, l_of_k_worst = find_lk_for_k(x, y, z, worst_h)
plot_intercept(name, x_intercept_worst, y_intercept_worst, worst_h, best=False)
plot_LK(name, k_worst, l_of_k_worst, worst_h, best=False)