from PH_method import *


@nb.njit
def generate_noise_trajectory(num_points):
    """
    Generate a 3D noise trajectory.

    Parameters:
    - num_points: Number of points in the trajectory.

    Returns:
    - Arrays for x, y, and z coordinates of the noise trajectory.
    """
    x = np.random.randn(num_points)
    y = np.random.randn(num_points)
    z = np.random.randn(num_points)
    return x, y, z

name = "noise"

#x_graph, y_graph, z_graph = generate_noise_trajectory(num_points=1000)
#plot_3D(name, x_graph, y_graph, z_graph)

x, y, z = generate_noise_trajectory(num_points=10000)

best_h = optimise_plane_lk(x, y, z, height_start=-2, height_stop=2)
x_intercept_best, y_intercept_best = find_intercepts(x, y, z, best_h) 
k_best, l_of_k_best = find_lk_for_k(x, y, z, best_h)
plot_intercept(name, x_intercept_best, y_intercept_best, best_h)
plot_LK(name, k_best, l_of_k_best, best_h)

worst_h = optimise_plane_lk(x, y, z, height_start=-2, height_stop=2, max_diff=False)
x_intercept_worst, y_intercept_worst = find_intercepts(x, y, z, worst_h) 
k_worst, l_of_k_worst = find_lk_for_k(x, y, z, worst_h)
plot_intercept(name, x_intercept_worst, y_intercept_worst, worst_h, best=False)
plot_LK(name, k_worst, l_of_k_worst, worst_h, best=False)