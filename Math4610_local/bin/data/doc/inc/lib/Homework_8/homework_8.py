import random
import numpy as np
import diagonal_matrix_generator, gauss_seidel, jacobi_iteration, lu_factorization, leslie_jacobi

# Parameters
n = 100
tolerance = 1e-10
max_iterations = 1000

# Generate a 100 by 100 diagonally dominant matrix and a random b vector
A = diagonal_matrix_generator.create_diagonally_dominant_matrix(n)
b = [random.random() for _ in range(n)]

### 1
# Apply the Jacobi iteration method
jacobi_iteration_solution, jacobi_count = jacobi_iteration.jacobi_iteration(A, b, tolerance, max_iterations)
print()
# Print the solution
print("Jacobi Solution:")
print(jacobi_iteration_solution)

### 2
# Apply the Jacobi iteration method to a leslie matrix with modified matrix-vector multiplications
#TODO this needs to be explained because you can't really do the jacobi iteration on a leslie matrix

# Define your Leslie matrix and initial population vector
# Modify these values according to your specific problem
leslie_matrix = np.array([[0, 2, 0.1],
                           [0.5, 0, 0],
                           [0, 0.4, 0]])

initial_population = np.array([100, 50, 25])

# Call the Jacobi iteration function to find the stable population distribution
stable_population = leslie_jacobi.leslie_jacobi_iteration(leslie_matrix, initial_population)

print()
# Print the result
print("Stable Population Distribution:")
print(stable_population)

### 3
print()
# Apply the Gauss-Seidel iteration method
gauss_seidel_solution, gs_count = gauss_seidel.gauss_seidel(A, b, tolerance, max_iterations)

# Print the solution
print("Gauss-Seidel Solution:")
print(gauss_seidel_solution)

### 4
# Compare LU Factorization, Jacobi, and Gauss-Seidel operations
# LU Factorization
L, U, count_lu = lu_factorization.lu_decomposition(A)

# Solving Ly = b
y, count_forward = lu_factorization.forward_substitution(L, b)

# Solving Ux = y
x, count_backward = lu_factorization.backward_substitution(U, y)

# Total operations
lu_count = count_lu + count_forward + count_backward

### 5
# Test code
print()
print("****** Operations Comparison Table ******")
print("-----------------------------------------")
print("Jacobi  | Gauss-Seidel | LU Factorization")
print("--------+--------------+-----------------")
print(f"{jacobi_count} | {gs_count}       | {lu_count}")
print("-----------------------------------------")
print()
