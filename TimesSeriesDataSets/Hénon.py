from PH_method import *


@nb.njit
def henon_map(x0=0.1, y0=0., z0=0., a=1.4, b=0.3, num_steps=10000):
    """
    Generate the trajectory of the Hénon map.
    
    Parameters:
    - x0, y0 (float): Initial conditions for the Hénon map.
    - a, b (float): Parameters of the Hénon map.
    - num_steps (int): Number of iterations.
    
    Returns:
    - x, y (numpy.ndarray): Arrays of x and y coordinates of the Hénon map trajectory.
    """
    x, y, z = np.zeros(num_steps + 1), np.zeros(num_steps + 1), np.zeros(num_steps + 1)
    x[0], y[0], z[0] = x0, y0, z0
    for i in range(num_steps):
        x[i + 1] = 1 - a * x[i] ** 2 + y[i]
        y[i + 1] = b * x[i]
        z[i + 1] = z[i]
    return x, y, z

name ="henon"

x_graph, y_graph, z_graph = henon_map(num_steps=1000)
plot_3D(name, x_graph, y_graph, z_graph, cmap=plt.cm.magma)