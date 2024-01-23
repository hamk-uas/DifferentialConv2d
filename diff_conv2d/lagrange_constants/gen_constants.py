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
def sample_poly_derivatives(N, x):
    return np.vectorize(lambda k: mp.ff(N - 1, k))(np.arange(N)).reshape(N, 1)*x**np.flip(np.arange(N)).reshape(N, 1)

# Calculation
x = np.arange(N_x, dtype=mp.mpf)
y = np.arange(N_y, dtype=mp.mpf)
# Samples of kth first derivatives of polynomials x^(N_x - 1) and y^(N_y - 1) with dimensions ("k_x", "x") or ("k_y", "y")
px_d = sample_poly_derivatives(N_x, x)
py_d = sample_poly_derivatives(N_y, x)
# Samples of kth first derivatives of 2-d polynomial x^(N_x - 1)y^(N_y - 1) with dimensions ("k_y", "k_x", "y", "x")
pxy_d = px_d.reshape(1, N_x, 1, N_x)*py_d.reshape(N_y, 1, N_y, 1)
# Flatten derivative order dimensions and flatten spatial dimensions
pxy_d_flat = pxy_d.reshape(N_y*N_x, N_y*N_x)
# Derivatives of 2-d polynomial at x_0, y_0
px_0_d = sample_poly_derivatives(N_x, x_0)
py_0_d = sample_poly_derivatives(N_y, y_0)
pxy_0_d = px_0_d.reshape(1, N_x)*py_0_d.reshape(N_y, 1)
pxy_0_d_flat = pxy_0_d.reshape(N_y, N_x, 1)
#c = mp.lu_solve(pxy_d_flat, pxy_0_d_flat)

#x = mp.arange(N)  # Sample positions (for x and y)
#z = x**(N-1) # Data, a polynomial
#A = hankel(y[0:N], y[N-1:N*2-1])  # Matrix of inputs to the prediction
#b = y[-N:]  # Prediction target vector
#c = mp.lu_solve(A, b)  # Least squares solve prediction coefficients
#return c # Round to integer

