from gen_constants import *

# Kernel size
N_y = 3
N_x = 3  

# Transformation from-coordinates 
y_0 = (N_y - 1)/2
x_0 = (N_x - 1)/2  

# Transformation to-coordinates
y_1 = (N_y - 1)/2 - 1
x_1 = (N_x - 1)/2 - 1

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

kernel = np.zeros((N_y, N_x))
kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
print_matrix(kernel, "kernel")
print()

transformed_kernel = np.array((mp.matrix(kernel.reshape((1, N_x*N_y)))*mp.matrix(transformation_matrix)).tolist()).reshape(N_y, N_x)
print_matrix(transformed_kernel, "transformed kernel")