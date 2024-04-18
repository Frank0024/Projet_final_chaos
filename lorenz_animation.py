# Tire de https://jakevdp.github.io/blog/2013/02/16/animating-the-lorentz-system-in-3d/

import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from matplotlib import animation
from collections import deque


def lorentz_deriv(X, t0, sigma=10., beta=8./3, rho=28.0):
    """Compute the time-derivative of a Lorentz system."""
    x, y, z = X
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

# parameters for the animation
step = 5            # how many steps to show per iteration
history_len = 1000  # how many trajectory points to display
plt.rcParams['figure.figsize'] = 8, 8
plt.rcParams['font.size'] = 18
plt.rcParams['lines.linewidth'] = 3

# Initial conditions
n_intitial_conditions = 35
x0 = np.zeros((n_intitial_conditions, 3))
x0[:, 1] = np.linspace(1, 1.0000000000024, n_intitial_conditions)

# Solve for the trajectories
t = np.linspace(0, 45, 45 * 100)
x_t = np.asarray([integrate.odeint(lorentz_deriv, x0i, t) for x0i in x0])

# Set up figure & 3D axis for animation
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.axis('off')

# setting up the lines/points to be plotted.
lines = []
pts = []
history_x = []
history_y = []
history_z = []
for c in x0:
    line, = ax.plot([], [], [], '-', lw=1, alpha=0.2)
    lines.append(line)
    pt, = ax.plot([], [], [], 'o', c="r", ms=2)
    pts.append(pt)
    history_x.append(deque(maxlen=history_len))
    history_y.append(deque(maxlen=history_len))
    history_z.append(deque(maxlen=history_len))

# prepare the axes limits
ax.set_xlim((-25, 25))
ax.set_ylim((-25, 25))
ax.set_zlim((5, 55))

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(15, 115)

# animation function.  This will be called sequentially with the frame number
def update(i):

    if i == 0:
        for line, pt, hx, hy, hz in zip(lines, pts, history_x, history_y, history_z):
            line.set_data_3d([], [], [])
            pt.set_data_3d([], [], [])
            hx.clear()
            hy.clear()
            hz.clear()

    for line, pt, xi, hx, hy, hz in zip(lines, pts, x_t, history_x, history_y, history_z):
        for j in range(step):
            x, y, z = xi[step * i + j].T
            hx.appendleft(x)
            hy.appendleft(y)
            hz.appendleft(z)
        line.set_data_3d(hx, hy, hz)
        pt.set_data_3d(x, y, z)

    if i == int(x_t.shape[1] / step - 1):
        for pt in pts:
            pt.set_data_3d([], [], [])

    fig.canvas.draw()
    return lines + pts

# instantiate the animator.
anim = animation.FuncAnimation(fig, update, int(x_t.shape[1] / step), interval=1, repeat=False)
plt.show()
