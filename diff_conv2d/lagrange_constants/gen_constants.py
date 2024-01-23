# Copyright 2024 HAMK Häme University of Applied Sciences
# Author: Olli Niemitalo (Olli.Niemitalo@hamk.fi)

import numpy as np
from scipy.linalg import hankel
import mpmath as mp
import json

mp.mp.prec = 500  # Precision in bits

# Input variables
N_x = 3  # Kernel size
N_y = 3  # Kernel size
x_0 = (N_x - 1)/2  # Transformation from-coordinates 
y_0 = (N_y - 1)/2  #
x_1 = (N_x - 1)/2 - 1  # Transformation to-coordinates
y_1 = (N_y - 1)/2 - 1  # 

# Sample kth first derivatives, 0 <= k < N, of 1-d polynomial x^(N - 1) at given x
# Returned data has dimensions ("k", "x")
def poly_derivatives(N, x):
    return np.vectorize(lambda k: mp.ff(N - 1, k))(np.arange(N)).reshape(N, 1)*x**np.flip(np.arange(N)).reshape(N, 1)

# Calculate assembled differential kernels spanning 0 <= x < N_x, 0 <= y < N_y that
# correspond to each 2-d derivative of polynomial data of x-degree N_x - 1 and y-degree N_y - 1,
# at x_0, y_0. We use the sign convention corresponding to multiplying an input data window with a kernel.
def diff_kernel(N_y, N_x, y_0, x_0):
  # Collect enough example windows of polynomial data to ensure unique solution
  pxy_flat_list = []
  pxy_0_d_flat_list = []
  for y_orig in mp.arange(N_y):
    for x_orig in mp.arange(N_x):
      # Polynomial values within the window
      x = np.arange(x_orig, x_orig + N_x, dtype=mp.mpf)
      y = np.arange(y_orig, y_orig + N_y, dtype=mp.mpf)
      px = x**(N_x - 1)
      py = y**(N_y - 1)
      pxy = px.reshape(1, N_x)*py.reshape(N_y, 1)
      pxy_flat = pxy.reshape(N_y*N_x)
      pxy_flat_list.append(pxy_flat)
      # Derivatives of 2-d polynomial at x_orig + x_0, y_orig + y_0 with window top left corner at x_orig, y_orig
      px_0_d = poly_derivatives(N_x, x_orig + x_0)
      py_0_d = poly_derivatives(N_y, y_orig + y_0)
      pxy_0_d = px_0_d.reshape(1, N_x)*py_0_d.reshape(N_y, 1)
      pxy_0_d_flat = pxy_0_d.reshape(N_y*N_x)
      pxy_0_d_flat_list.append(pxy_0_d_flat)

  pxy_flat = np.array(pxy_flat_list)
  pxy_0_d_flat = np.array(pxy_0_d_flat_list)

  c_list = []
  for k in range(N_x*N_y):
    c_list.append(mp.lu_solve(pxy_flat, pxy_0_d_flat[:,k]))

  c = np.array(c_list)
  return c

def print_matrix(c, name):
  print(f"{name} = ")
  print(np.array(mp.chop(np.array(c), tol=0.5**(mp.mp.prec//2))).astype(np.float64))

print(f"N_y, N_x = {N_y}, {N_x}")
print(f"x_0, y_0 = {x_0}, {y_0}")
print(f"x_1, y_1 = {x_1}, {y_1}")
print()

c_0 = diff_kernel(N_y, N_x, y_0, x_0)
c_1 = diff_kernel(N_y, N_x, y_1, x_1)
print_matrix(c_0, "c_0")
print()

print_matrix(c_1, "c_1")
print()

c_0_inv = np.array((mp.matrix(c_0)**-1).tolist())
print_matrix(c_0_inv, "c_0_inv")
print()

transformation_matrix = np.array((mp.matrix(c_0_inv)*mp.matrix(c_1)).tolist())
print_matrix(transformation_matrix, "flattended transformation matrix")
#print_matrix(transformation_matrix.reshape((N_y, N_x, N_y, N_x)), "transformation matrix")
print()

kernel = np.ones((N_y, N_x))
print_matrix(kernel, "kernel")
print()

transformed_kernel = np.array((mp.matrix(kernel.reshape((1, N_x*N_y)))*mp.matrix(transformation_matrix)).tolist()).reshape(N_y, N_x)
print_matrix(transformed_kernel, "transformed kernel")
