"""
===========================
The double pendulum problem
===========================

This animation illustrates the double pendulum problem.

Double pendulum formula translated from the C code at
http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c
http://www.physics.usyd.edu.au/~wheat/dpend_html/

Adapted and downloaded from: https://matplotlib.org/stable/gallery/animation/double_pendulum.html

For the math, see https://physicspython.wordpress.com/tag/double-pendulum/
                  https://scipython.com/blog/the-double-pendulum/
"""

import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from matplotlib import animation
from collections import deque


# parameters for the dynamics
G = 9.8      # acceleration due to gravity, in m/s^2
L1 = 1.0     # length of pendulum 1 in m
L2 = 1.0     # length of pendulum 2 in m
L = L1 + L2  # maximal length of the combined pendulum
M1 = 1.0     # mass of pendulum 1 in kg
M2 = 1.0     # mass of pendulum 2 in kg

# parameters for the animation
dt = 0.02          # time step for numerical integration
t_stop = 50        # how many seconds to simulate
history_len = 50  # how many trajectory points to display
plt.rcParams['figure.figsize'] = 8, 8
plt.rcParams['font.size'] = 18
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False



def vector_field(state, t):
    """Vector field of the dynamics."""

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * np.cos(delta) * np.cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * np.sin(delta) * np.cos(delta)
                + M2 * G * np.sin(state[2]) * np.cos(delta)
                + M2 * L2 * state[3] * state[3] * np.sin(delta)
                - (M1+M2) * G * np.sin(state[0]))
               / den1)

    dydx[2] = state[3]

    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * np.sin(delta) * np.cos(delta)
                + (M1+M2) * G * np.sin(state[0]) * np.cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * np.sin(delta)
                - (M1+M2) * G * np.sin(state[2]))
               / den2)

    return dydx


def get_trajectory(initial_condition):
    """Numerical integration of the dynamics from an initial consition."""

    # create a time array from 0..t_stop sampled at 0.02 second steps
    t = np.arange(0, t_stop, dt)

    # integrate your ODE unp.sing scipy.integrate.
    y = integrate.odeint(vector_field, initial_condition, t)

    # converts polar to cartesian coordinates
    x1 = L1*np.sin(y[:, 0])
    y1 = -L1*np.cos(y[:, 0])

    x2 = L2*np.sin(y[:, 2]) + x1
    y2 = -L2*np.cos(y[:, 2]) + y1

    return x1, y1, x2, y2


# initial conditions
#   th1 and th2 are the initial angles (degrees)
#   w10 and w20 are the initial angular velocities (degrees per second)
#   initial condition given in format [th1, w1, th2, w2]
initial_conditions = []
initial_conditions.append(np.radians([179.999, 0.0, 180.000000000000, 0.0]))
initial_conditions.append(np.radians([179.999, 0.0, 180.000000000001, 0.0]))
initial_conditions.append(np.radians([179.999, 0.0, 180.000000000002, 0.0]))
initial_conditions.append(np.radians([179.999, 0.0, 180.000000000003, 0.0]))


# computes the trajectories by numerical integration
trajectories = []
for initial_condition in initial_conditions:
    trajectories.append(get_trajectory(initial_condition))


# creates the plotting objects
L = 1.1 * L
fig, ax = plt.subplots(subplot_kw=dict(autoscale_on=False, xlim=(-L, L), ylim=(-L, L)))
ax.set_aspect('equal')

lines = []
traces = []
history_x = []
history_y = []
for initial_condition in initial_conditions:
    line, = ax.plot([], [], 'o-', lw=3)
    lines.append(line)
    trace, = ax.plot([], [], '.-', lw=2, ms=5)
    traces.append(trace)
    history_x.append(deque(maxlen=history_len))
    history_y.append(deque(maxlen=history_len))
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def update(i):

    for l in range(len(initial_conditions)):

        thisx = [0, trajectories[l][0][i], trajectories[l][2][i]]
        thisy = [0, trajectories[l][1][i], trajectories[l][3][i]]

        if i == 0:
            history_x[l].clear()
            history_y[l].clear()

        history_x[l].appendleft(thisx[2])
        history_y[l].appendleft(thisy[2])

        lines[l].set_data(thisx, thisy)
        traces[l].set_data(history_x[l], history_y[l])

    time_text.set_text(time_template % (i*dt))

    return lines + traces + [time_text]


ani = animation.FuncAnimation(fig, update, len(trajectories[0][0]), interval=dt*1000, blit=True)
plt.show()
