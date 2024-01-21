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
  if np.max(np.abs(c)) > 9007199254740992.0:  # See https://stackoverflow.com/a/1848953/4770915
    print ("Warning: Too large coefficient for datatype double")
  c_rounded = c.apply(mp.nint)
  if np.any(np.abs(c - c_rounded) > 0.0000000001):
    print ("Warning: Dubious precision")
  return c_rounded # Round to integer

M = 26 # Maximum kernel size

c_list_list = []
for N in range(1, M+1):
  c_list = []
  for s in range(1, (M+1)//2):
    c_list.append(list(np.array(lagrange_padding_coefs(N, s), dtype=np.double)))
  c_list_list.append(c_list)

json_string = json.dumps(c_list_list).replace("]], [[", "]],\n [[").replace("], [", "],\n  [").replace("]]]", "]]\n]")

with open("lagrange_padding_coefs.json", "w") as out_file:
  out_file.write(json_string)