import numpy as np
from scipy.linalg import hankel
import mpmath as mp
import json

mp.mp.prec = 500  # Precision in bits

# Get coefficients for polynomial padding
# N = Number of linear prediction coefficients
# s = How much forward to predict after the last input
def lagrange_padding_coefs(N, s):
  L = N*2-1+s  # Data length, long enough to form large enough a system to solve and to include prediction targets
  x = np.arange(L, dtype=mp.mpf)  # Sample positions
  y = x**(N-1) # Data, a polynomial
  A = hankel(y[0:N], y[N-1:N*2-1])  # Matrix of inputs to the prediction
  b = y[-N:]  # Prediction target vector
  c = mp.lu_solve(A, b)  # Least squares solve prediction coefficients
  c_rounded = c.apply(mp.nint)
  if np.any(np.abs(c - c_rounded) > 0.0000000001):
    raise RuntimeError(f"Dubious precision in {np.abs(c - c_rounded)}")
  residual = b - np.array(mp.matrix(A)*c_rounded)
  if np.max(np.abs(residual)) > 0:
    raise RuntimeError(f"Nonzero prediction residual {np.max(np.abs(residual))}")
  if np.max(np.abs(np.array(c_rounded).astype(np.float64) - c_rounded)) > 0:
    raise RuntimeError(f"Too large value {np.max(np.abs(c_rounded))} to be represented exactly as float64")
  return c_rounded # Round to integer

M = 27 # Maximum kernel size

c_matrix_list = []
for N in range(1, M+1):
  c_matrix = []
  for s in range(1, (M+1)//2):
    c_matrix.append(list(np.array(lagrange_padding_coefs(N, s), dtype=np.float64)))
  c_matrix_list.append(c_matrix)

json_string = json.dumps(c_matrix_list).replace("]], [[", "]],\n [[").replace("], [", "],\n  [").replace("]]]", "]]\n]")

with open("lagrange_padding_coefs.json", "w") as out_file:
  out_file.write(json_string)
