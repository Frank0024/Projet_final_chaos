import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numba as nb
from sklearn.metrics import r2_score
import seaborn as sns
import os

PALETTEB = sns.color_palette("bright")
PALETTEC = sns.color_palette("colorblind")
PALETTEP = sns.color_palette("pastel")

@nb.njit
def find_intercepts(x, y, z, plane_height):
    """
    Identify the intercepts where a 3D trajectory crosses a horizontal plane at a specified height.
    
    Parameters:
    - x (array of float): The x-coordinates of the trajectory.
    - y (array of float): The y-coordinates of the trajectory.
    - z (array of float): The z-coordinates of the trajectory, representing vertical position.
    - plane_height (float): The height of the plane to check for intercepts.
    
    Returns:
    - tuple of arrays: Two arrays containing the x and y coordinates of the points where the trajectory crosses the specified height.
    """
    x_intercept = []
    y_intercept = []
    for i in range(len(z) - 1):
        if z[i] < plane_height and z[i + 1] > plane_height:
            x_intercept.append(x[i])
            y_intercept.append(y[i])
    return np.array(x_intercept), np.array(y_intercept)

@nb.njit
def find_intercepts_2d(x, y, plane_height):
    """
    Identify the intercepts where a 3D trajectory crosses a horizontal plane at a specified height.
    
    Parameters:
    - x (array of float): The x-coordinates of the trajectory.
    - y (array of float): The y-coordinates of the trajectory.
    - plane_height (float): The height of the plane to check for intercepts.
    
    Returns:
    - tuple of arrays: Two arrays containing the x and y coordinates of the points where the trajectory crosses the specified height.
    """
    x_intercept = []
    y_intercept = []
    for i in range(len(y) - 1):
        if y[i] < plane_height and y[i + 1] > plane_height:
            x_intercept.append(x[i])
            y_intercept.append(y[i])
    return np.array(x_intercept), np.array(y_intercept)

@nb.njit
def smk(x, y, m, k):
    """
    Generate a subset of points from a 2D trajectory using a sliding window approach.
    
    Parameters:
    - x (array of float): The x-coordinates of the trajectory.
    - y (array of float): The y-coordinates of the trajectory.
    - m (int): The size of the sliding window.
    - k (int): The step size between successive windows.
    
    Returns:
    - ndarray: A 2D NumPy array containing points sampled according to the sliding window parameters.
    """
    N = len(x)
    num_points = int(np.floor((N - m) / k))
    s = np.empty((num_points, 2))  # Pre-allocate a NumPy array
    for i in range(num_points):
        index = m + (i * k) - 1
        s[i, 0] = x[index]
        s[i, 1] = y[index]
    return s

@nb.njit
def lmk(x, y, m, k):
    """
    Calculate the approximate length of segments of a 2D trajectory using a sliding window.
    
    Parameters:
    - x (array of float): The x-coordinates.
    - y (array of float): The y-coordinates.
    - m (int): The size of the sliding window.
    - k (int): The step size between successive windows.
    
    Returns:
    - float: The scaled average distance covered in each segment of the trajectory.
    """
    points = smk(x, y, m, k)
    total_distance = 0.0
    for i in range(len(points) - 1):
        total_distance += np.linalg.norm(points[i + 1] - points[i])
    if len(points) > 1:
        return (total_distance * (len(x) - m)) / (k * (len(points) - 1))
    else:
        return 0.0


@nb.njit
def lk(x, y, k):
    """
    Compute the average length of a 2D trajectory using multiple sliding windows to enhance accuracy.
    
    Parameters:
    - x (array of float): The x-coordinates.
    - y (array of float): The y-coordinates.
    - k (int): The number of sliding windows to apply.
    
    Returns:
    - float: The average length computed over all specified window configurations.
    """
    moyenne = 0.0
    for m in range(1, k + 1):
        moyenne += lmk(x, y, m, k)
    return moyenne / k

@nb.njit
def find_lk_for_k(x, y, z, height, ks=np.arange(1, 20)):
    """
    Apply the Lk method to compute the trajectory length for various configurations of sliding window sizes, 
    evaluated at the intercepts with a specified horizontal plane.
    
    Parameters:
    - x (array of float): The x-coordinates of the trajectory.
    - y (array of float): The y-coordinates of the trajectory.
    - z (array of float): The z-coordinates of the trajectory.
    - height (float): The height of the plane to find intercepts.
    - ks (array of int, optional): The range of k-values (window sizes) to use.
    
    Returns:
    - tuple: Two arrays, the first containing the k-values used, and the second containing the logarithm of the Lk lengths.
    """
    x_intercept, y_intercept = find_intercepts(x, y, z, height)
    l_de_k = np.zeros(len(ks), dtype=np.float64)
    for i in range(len(ks)):
        l_de_k[i] = lk(x_intercept, y_intercept, ks[i])
    return ks, np.log(l_de_k)

@nb.njit
def average_difference(points):
    """
    Calculate the average difference between consecutive values in a 1D list using Numba for performance optimization.

    Parameters:
    - points (array of float/int): Array of numeric values.

    Returns:
    - float: The average difference between each successive value.
    """
    n = len(points)
    if n < 2:
        return 0.0  # Return 0 if there are not enough points to compare

    total_difference = 0.0
    # Loop through the array of points and calculate the difference between consecutive points
    for i in range(1, n):
        difference = abs(points[i] - points[i-1])
        total_difference += difference

    # Calculate the average difference
    average_diff = total_difference / (n - 1)
    return average_diff


def optimise_plane_lk(x, y, z, height_start=5, height_stop=25, step=1, max_diff=True):
    """
    Optimise the plane height to find the height that maximizes the average difference 
    in the logarithm of the trajectory lengths calculated by the Lk method.

    This function iterates through a range of plane heights and determines the height
    at which the average difference of the logarithm of lengths of trajectories, 
    intercepted at each height, is maximized.

    Parameters:
    - x (array of float): The x-coordinates of the trajectory.
    - y (array of float): The y-coordinates of the trajectory.
    - z (array of float): The z-coordinates of the trajectory.
    - height_start (int, optional): The starting height of the plane.
    - height_stop (int, optional): The ending height of the plane.
    - step (int, optional): The increment step between heights.

    Returns:
    - float: The height that maximizes the average difference of the logarithmic lengths.
    """
    possible_height = np.arange(height_start, height_stop, step)

    if max_diff == True :
        max_moy = 0
        best_height = 0
        for h in possible_height:
            lk_of_k = find_lk_for_k(x, y, z, h)[1]
            moyenne_dy = average_difference(lk_of_k)
            if moyenne_dy > max_moy:
                max_moy = moyenne_dy
                best_height = h
    elif max_diff == False:
        max_moy = np.inf
        best_height = 0
        for h in possible_height:
            lk_of_k = find_lk_for_k(x, y, z, h)[1]
            moyenne_dy = average_difference(lk_of_k)
            if moyenne_dy < max_moy:
                max_moy = moyenne_dy
                best_height = h
    return best_height

@nb.njit
def rk4_step(x, y, z, dt, function):
    """
    Perform one step of the Runge-Kutta 4th order method for solving a system of ODEs.

    Parameters:
    - x, y, z: Arrays of x, y, and z coordinates of the trajectory.
    - dt: Time step.

    Returns:
    - x_new, y_new, z_new: Arrays of new x, y, and z coordinates of the trajectory.
    """
    k1_x, k1_y, k1_z = function(x, y, z)
    k2_x, k2_y, k2_z = function(x + 0.5 * dt * k1_x, y + 0.5 * dt * k1_y, z + 0.5 * dt * k1_z)
    k3_x, k3_y, k3_z = function(x + 0.5 * dt * k2_x, y + 0.5 * dt * k2_y, z + 0.5 * dt * k2_z)
    k4_x, k4_y, k4_z = function(x + dt * k3_x, y + dt * k3_y, z + dt * k3_z)
    x_new = x + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
    y_new = y + (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
    z_new = z + (dt / 6.0) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)
    return x_new, y_new, z_new
