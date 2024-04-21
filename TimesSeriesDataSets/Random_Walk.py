from PH_method import *

@nb.njit
def random_walk_3d(num_steps):
    """
    Generate a 3D random walk trajectory.

    Parameters:
    - num_steps: Number of steps in the random walk.

    Returns:
    - Arrays for x, y, and z coordinates of the random walk trajectory.
    """
    # Generate random steps in each dimension
    x_steps = np.random.normal(0, 1, num_steps)
    y_steps = np.random.normal(0, 1, num_steps)
    z_steps = np.random.normal(0, 1, num_steps)

    # Calculate cumulative sum to get the trajectory
    x_trajectory = np.cumsum(x_steps)
    y_trajectory = np.cumsum(y_steps)
    z_trajectory = np.cumsum(z_steps)

    return x_trajectory, y_trajectory, z_trajectory

name ="randomwalk"

x_graph, y_graph, z_graph = random_walk_3d(num_steps=100000)
plot_3D(name, x_graph, y_graph, z_graph, cmap=plt.cm.inferno)

x, y, z = random_walk_3d(num_steps=100000)

best_h = optimise_plane_lk(x, y, z, height_start=-20, height_stop=20, step=0.1)
x_intercept_best, y_intercept_best = find_intercepts(x, y, z, best_h) 
k_best, l_of_k_best = find_lk_for_k(x, y, z, best_h)
plot_intercept(name, x_intercept_best, y_intercept_best, best_h)
plot_LK(name, k_best, l_of_k_best, best_h)

worst_h = optimise_plane_lk(x, y, z, height_start=-20, height_stop=20, step=0.1, max_diff=False)
x_intercept_worst, y_intercept_worst = find_intercepts(x, y, z, worst_h) 
k_worst, l_of_k_worst = find_lk_for_k(x, y, z, worst_h)
plot_intercept(name, x_intercept_worst, y_intercept_worst, worst_h, best=False)
plot_LK(name, k_worst, l_of_k_worst, worst_h, best=False)