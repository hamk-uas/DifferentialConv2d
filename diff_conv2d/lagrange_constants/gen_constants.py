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
x_1 = 2  # Transformation to-coordinates

# Sample kth first derivatives, 0 <= k < N, of polynomial x^(N - 1) at given x
# Returned data has dimensions ("k", "x")
def poly_derivatives(N, x):
    return np.vectorize(lambda k: mp.ff(N - 1, k))(np.arange(N)).reshape(N, 1)*x**np.flip(np.arange(N)).reshape(N, 1)

pxy_flat_list = []
pxy_0_d_flat_list = []
for y_orig in mp.arange(N_x, N_x + N_y):
  for x_orig in mp.arange(N_x):
    # Polynomial values within the window
    x = np.arange(x_orig, x_orig + N_x, dtype=mp.mpf)
    y = np.arange(y_orig, y_orig + N_y, dtype=mp.mpf)
    px = x**(N_x - 1)
    py = y**(N_y - 1)
    pxy = px.reshape(1, N_x)*py.reshape(N_y, 1)
    pxy_flat = pxy.reshape(N_y*N_x)
    pxy_flat_list.append(pxy_flat)
    # Derivatives of 2-d polynomial at x_0, y_0 referenced to window origin
    px_0_d = poly_derivatives(N_x, x_orig + x_0)
    py_0_d = poly_derivatives(N_y, y_orig + y_0)
    pxy_0_d = px_0_d.reshape(1, N_x)*py_0_d.reshape(N_y, 1)
    pxy_0_d_flat = pxy_0_d.reshape(N_y*N_x)
    pxy_0_d_flat_list.append(pxy_0_d_flat)

pxy_flat = np.array(pxy_flat_list)
pxy_0_d_flat = np.array(pxy_0_d_flat_list)

mp.chop(mp.lu_solve(pxy_flat, pxy_0_d_flat[:,4]))

#x = mp.arange(N)  # Sample positions (for x and y)
#z = x**(N-1) # Data, a polynomial
#A = hankel(y[0:N], y[N-1:N*2-1])  # Matrix of inputs to the prediction
#b = y[-N:]  # Prediction target vector
#c = mp.lu_solve(A, b)  # Least squares solve prediction coefficients
#return c # Round to integer

