from PH_method import *


def brownian_motion(x0=0., y0=0., z0=0., dimension=3, num_steps=1000000, dt=0.1):
    x = np.zeros(num_steps + 1)
    y = np.zeros(num_steps + 1)
    z = np.zeros(num_steps + 1)
    x[0], y[0], z[0] = x0, y0, z0

    for j in range(1, num_steps + 1):
        s = np.sqrt(6 * dt) * np.random.randn(1)
        dx = np.random.randn()
        dy = np.random.randn()
        if dimension == 3:
            dz = np.random.randn()
            norm = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            dz *= s / norm
            z[j] = z[j - 1] + dz
        if dimension == 2:
            norm = np.sqrt(dx ** 2 + dy ** 2)
            z[j] = 0
        dx *= s / norm
        dy *= s / norm
        x[j] = x[j - 1] + dx
        y[j] = y[j - 1] + dy

    return x, y, z

name ="brownian"

#x_graph, y_graph, z_graph = brownian_motion(num_steps=100000)
#plot_3D(name, x_graph, y_graph, z_graph, cmap=plt.cm.rainbow)

#x_graph_2D, y_graph_2D, z_graph_2D = brownian_motion(dimension=2, num_steps=100000)
#plot_2D(name, x_graph_2D, y_graph_2D, cmap=plt.cm.rainbow)

x_2D, y_2D, z_2D = brownian_motion(num_steps=1000000, dimension=2)
best_y = optimise_line_y(x_2D, y_2D, value_start=-10, value_stop=10, step=0.1)
k_best_y, ln_of_k_best_2D = find_lk_for_k_2d(x_2D, y_2D, best_y)
plot_LK_2D(name, k_best_y, ln_of_k_best_2D, best_y)
worst_y = optimise_line_y(x_2D, y_2D, value_start=-10, value_stop=10, step=0.1, max_diff=False)
k_worst_y, ln_of_k_worst_2D = find_lk_for_k_2d(x_2D, y_2D, worst_y)
plot_LK_2D(name, k_worst_y, ln_of_k_worst_2D, worst_y, best=False)

x, y, z = brownian_motion(num_steps=1000000)
best_h = optimise_plane_lk(x, y, z, height_start=-10, height_stop=10, step=0.1)
x_intercept_best, y_intercept_best = find_intercepts(x, y, z, best_h) 
k_best, l_of_k_best = find_lk_for_k(x, y, z, best_h)
plot_intercept(name, x_intercept_best, y_intercept_best, best_h)
plot_LK(name, k_best, l_of_k_best, best_h)

worst_h = optimise_plane_lk(x, y, z, height_start=-10, height_stop=10, step=0.1, max_diff=False)
x_intercept_worst, y_intercept_worst = find_intercepts(x, y, z, worst_h) 
k_worst, l_of_k_worst = find_lk_for_k(x, y, z, worst_h)
plot_intercept(name, x_intercept_worst, y_intercept_worst, worst_h, best=False)
plot_LK(name, k_worst, l_of_k_worst, worst_h, best=False)
