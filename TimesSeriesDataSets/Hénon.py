from PH_method import *


@nb.njit
def henon_map(x0=0.1, y0=0.0, a=1.4, b=0.3, num_steps=10000):
    """
    Generate the trajectory of the Hénon map.
    
    Parameters:
    - x0, y0 (float): Initial conditions for the Hénon map.
    - a, b (float): Parameters of the Hénon map.
    - num_steps (int): Number of iterations.
    
    Returns:
    - x, y (numpy.ndarray): Arrays of x and y coordinates of the Hénon map trajectory.
    """
    x, y = np.zeros(num_steps + 1), np.zeros(num_steps + 1)
    x[0], y[0] = x0, y0
    for i in range(num_steps):
        x[i + 1] = 1 - a * x[i] ** 2 + y[i]
        y[i + 1] = b * x[i]
    return x, y

name ="henon"

#x_graph, y_graph = henon_map(num_steps=10000)
#plot_point_2D(name, x_graph, y_graph, cmap=plt.cm.magma)

x, y = henon_map(num_steps=100000)
best_y = optimise_line_y(x, y, value_start=-0.3, value_stop=0.3, step=0.05)
k_best_y, ln_of_k_best_2D = find_lk_for_k_2d(x, y, best_y)
plot_LK_2D(name, k_best_y, ln_of_k_best_2D, best_y)
worst_y = optimise_line_y(x, y, value_start=-0.3, value_stop=0.3, step=0.05, max_diff=False)
k_worst_y, ln_of_k_worst_2D = find_lk_for_k_2d(x, y, worst_y)
plot_LK_2D(name, k_worst_y, ln_of_k_worst_2D, worst_y, best=False)