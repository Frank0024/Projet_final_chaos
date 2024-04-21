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

def plot_3D(name, x, y, z, cmap=plt.cm.viridis):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = cmap(np.linspace(0, 1, len(x)))

    # Tracer les trajectoires avec un gradient de couleur
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], lw=0.5)
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.set_zlabel('z', fontsize=16)

    out_dir = os.path.join("output", "01_3D_plot")
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(os.path.join(out_dir, name + "3D.png"), transparent=True, bbox_inches="tight", dpi=350)
    plt.close()

def plot_intercept(name, x_intercept, y_intercept, height, best=True):
    plt.scatter(x_intercept, y_intercept, s=5, marker="o", color=PALETTEB[0], label=f"Plan Z = {height:.2f}")
    plt.xlabel("x intercept", fontsize=16)
    plt.ylabel("y intercept", fontsize=16)
    plt.legend(fontsize=14)
    plt.minorticks_on()
    plt.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=14)

    if best:
        out_dir = os.path.join("output", "02_best_intercept")
        subname = "_best_intercept.png"
    else:
        out_dir = os.path.join("output", "03_worst_intercept")
        subname = "_worst_intercept.png"

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, name + subname), transparent=True, bbox_inches="tight")
    plt.close()

def plot_LK(name, k, l_of_k, height, best=True):
    plt.plot(k, l_of_k, linestyle="-", color=PALETTEB[2], alpha=0.4)
    plt.plot(k, l_of_k, marker='o', linestyle='', color=PALETTEB[2], label=f"Plan Z = {height:.2f}")
    plt.xlabel("k Values", fontsize=16)
    plt.ylabel("Log of Lk", fontsize=16)
    plt.legend(fontsize=14)
    plt.minorticks_on()
    plt.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=14)

    if best:
        out_dir = os.path.join("output", "04_best_LK")
        subname = "_best_LK.png"
    else:
        out_dir = os.path.join("output", "05_worst_LK")
        subname = "_worst_LK.png"

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, name + subname), transparent=True, bbox_inches="tight")
    plt.close()

x_graph, y_graph, z_graph = solve_lorenz(num_steps=100000)
plot_3D("lorenz", x_graph, y_graph, z_graph)

x, y, z = solve_lorenz(num_steps=1000000)

best_h = optimise_plane_lk(x, y, z)
x_intercept_best, y_intercept_best = find_intercepts(x, y, z, best_h) 
k_best, l_of_k_best = find_lk_for_k(x, y, z, best_h)
plot_intercept("lorenz", x_intercept_best, y_intercept_best, best_h)
plot_LK("lorenz", k_best, l_of_k_best, best_h)

worst_h = optimise_plane_lk(x, y, z, max_diff=False)
x_intercept_worst, y_intercept_worst = find_intercepts(x, y, z, worst_h) 
k_worst, l_of_k_worst = find_lk_for_k(x, y, z, worst_h)
plot_intercept("lorenz", x_intercept_worst, y_intercept_worst, worst_h, best=False)
plot_LK("lorenz", k_worst, l_of_k_worst, worst_h, best=False)
