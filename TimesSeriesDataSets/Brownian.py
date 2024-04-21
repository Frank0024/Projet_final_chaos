from PH_method import *


def brownian_motion(x0=0., y0=0., z0=0., num_steps=1000000, dt=0.1):
    x = np.zeros(num_steps + 1)
    y = np.zeros(num_steps + 1)
    z = np.zeros(num_steps + 1)

    x[0], y[0], z[0] = x0, y0, z0

    for j in range(1, num_steps + 1):
        s = np.sqrt(6 * dt) * np.random.randn(1)
        dx = np.random.randn()
        dy = np.random.randn()
        dz = np.random.randn()
        norm_dxyz = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        dx *= s / norm_dxyz
        dy *= s / norm_dxyz
        dz *= s / norm_dxyz
        x[j] = x[j - 1] + dx
        y[j] = y[j - 1] + dy
        z[j] = z[j - 1] + dz

    return x, y, z

name ="brownian"

x_graph, y_graph, z_graph = brownian_motion(num_steps=100000)
plot_3D(name, x_graph, y_graph, z_graph, cmap=plt.cm.rainbow)

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
