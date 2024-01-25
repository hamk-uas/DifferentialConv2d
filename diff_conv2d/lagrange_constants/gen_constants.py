# Copyright 2024 HAMK Häme University of Applied Sciences
# Author: Olli Niemitalo (Olli.Niemitalo@hamk.fi)

import numpy as np
import mpmath as mp
import json

mp.mp.prec = 250  # Precision in bits

# Sample kth first derivatives, 0 <= k < N, of 1-d polynomial x^(N - 1) at given x
# Returned data has dimensions ("k", "x")
def poly_derivatives(N, x):
    return np.vectorize(lambda k: mp.ff(N - 1, k))(np.arange(N)).reshape(N, 1)*mp.mpf(x)**np.flip(np.arange(N)).reshape(N, 1)

# Calculate assembled differential kernels spanning 0 <= y < N_y, 0 <= x < N_x that
# correspond to the values of N_y*N_x earliest 2-d derivatives of polynomial data of y-degree N_y - 1 and x-degree N_x - 1
# at y_0, x_0. We use the sign convention corresponding to multiplying an input data window with a kernel.
def diff_kernel(N_y, N_x, y_0, x_0):
  # Collect enough example windws of polynomial data to ensure unique solution
  pyx_flat_list = []
  pyx_0_d_flat_list = []
  for y_orig in mp.arange(N_y):
    for x_orig in mp.arange(N_x):
      # Polynomial values within the window
      x = np.arange(x_orig, x_orig + N_x, dtype=mp.mpf)
      y = np.arange(y_orig, y_orig + N_y, dtype=mp.mpf)
      py = y**(N_y - 1)
      px = x**(N_x - 1)
      pyx = py.reshape(N_y, 1)*px.reshape(1, N_x)
      pyx_flat = pyx.reshape(N_y*N_x)
      pyx_flat_list.append(pyx_flat)
      # Derivatives of 2-d polynomial at y_orig + y_0, x_orig + x_0 with window top left corner at y_orig, x_orig
      py_0_d = poly_derivatives(N_y, y_orig + y_0)
      px_0_d = poly_derivatives(N_x, x_orig + x_0)
      pyx_0_d = py_0_d.reshape(N_y, 1)*px_0_d.reshape(1, N_x)
      pyx_0_d_flat = pyx_0_d.reshape(N_y*N_x)
      pyx_0_d_flat_list.append(pyx_0_d_flat)

  pyx_flat = np.array(pyx_flat_list)
  pyx_0_d_flat = np.array(pyx_0_d_flat_list)

  c_list = []
  for k in range(N_y*N_x):
    c_list.append(mp.lu_solve(pyx_flat, pyx_0_d_flat[:,k]))

  c = np.array(c_list)
  return c

# Running this python file will in stand-alone fashion will generate transformation matrix files
if __name__ == "__main__":

  # Generate kernel transformation matrixes for this shared range of kernel height and width
  min_N = 1
  max_N = 7
  
  # Symmetric only?
  symmetric_only = True

  for N_y in range(min_N, max_N + 1):
    y_0 = (N_y - 1)/2
    for N_x in range(min_N, max_N + 1):
      if symmetric_only and N_x != N_y:
        continue
      print(f"Solving kernel size {N_y}x{N_x}")
      x_0 = (N_x - 1)/2
      c_0 = diff_kernel(N_y, N_x, y_0, x_0)
      c_0_inv = np.array((mp.matrix(c_0)**-1).tolist())
      transformation_matrix_matrix = []
      for y_1 in list(-0.5*(N_y%2 == 0) + y_0 + np.arange(-(N_y - 1)/2, (N_y - 1)/2 + 1 + (N_y%2 == 0))):
        transformation_matrix_list = []
        for x_1 in list(-0.5*(N_x%2 == 0) + x_0 + np.arange(-(N_x - 1)/2, (N_x - 1)/2 + 1 + (N_x%2 == 0))):
          print(f"Solving kernel size {N_y}x{N_x}, diff operator shift ({y_1-y_0}, {x_1-x_0}).")
          c_1 = diff_kernel(N_y, N_x, y_1, x_1)
          transformation_matrix = mp.matrix(c_0_inv)*mp.matrix(c_1)
          transformation_matrix = mp.chop(transformation_matrix, 0.5**(mp.mp.prec//2))
          transformation_matrix = np.array(transformation_matrix.tolist())
          float64_transformation_matrix = transformation_matrix.astype(np.float64)
          maxAbsError = np.max(np.abs(float64_transformation_matrix - transformation_matrix))
          if maxAbsError > 0.5**(mp.mp.prec//2):
            print(f"Lost precision in conversion to np.float64, maxAbsError = {maxAbsError}")
          transformation_matrix = float64_transformation_matrix
          transformation_matrix = np.transpose(transformation_matrix)
          int64_transformation_matrix = transformation_matrix.astype(np.int64)
          if (int64_transformation_matrix == transformation_matrix).all():
            transformation_matrix = int64_transformation_matrix
          transformation_matrix_list.append(transformation_matrix.tolist())
        transformation_matrix_matrix.append(transformation_matrix_list)
      out_filename = f"kernel_transformation_matrix_{N_y}x{N_x}.json"
      with open(out_filename, "w") as out_file:
        json_str = json.dumps(transformation_matrix_matrix)
        out_file.write(json_str)
      print(f"Wrote {out_filename}")