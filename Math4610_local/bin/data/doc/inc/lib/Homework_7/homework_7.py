import inverse_power_method, leslie, partitions, shifted_inverse_power_method
from vector_functions import power_method

A = [[4, 2], [1, 3]]
n = [1, 0]
tol = 1e-6
max_iter = 1000

max_value = power_method.power_method(A, n, tol, max_iter)
print("Largest Eigenvalue:", max_value)

smallest_eigenvalue = inverse_power_method.inverse_power_method(A, n, tol, max_iter)
print("Smallest Eigenvalue:", smallest_eigenvalue)


leslie_matrix = [[0.1, 2.0, 1.5, 0.5], [0.5, 0, 0, 0], [0, 0.7, 0, 0], [0, 0, 0.9, 0]]
n = [1, 1, 1, 1]
tol = 1e-6
max_iter = 1000

max_value = leslie.leslie_power_method(leslie_matrix, n, tol, max_iter)
print("Largest Eigenvalue of Leslie:", max_value)

# Estimate smallest and largest eigenvalues
smallest_eigenvalue = inverse_power_method.inverse_power_method(A, n, tol, max_iter)
largest_eigenvalue = power_method.power_method(A, n, tol, max_iter)

# Partition the interval
num_partitions = 10
partition_size = (largest_eigenvalue - smallest_eigenvalue) / num_partitions
eigenvalues = []

for i in range(num_partitions):
    shift = smallest_eigenvalue + i * partition_size + partition_size / 2
    eigenvalue = partitions.power_method_shifted(A, n, shift, tol, max_iter)
    eigenvalues.append(eigenvalue)

print("Eigenvalues near the partitions:", eigenvalues)

# Estimate smallest and largest eigenvalues
smallest_eigenvalue = inverse_power_method.inverse_power_method(A, n, tol, max_iter)
largest_eigenvalue = power_method.power_method(A, n, tol, max_iter)

# Calculate the shift (midpoint)
shift = (smallest_eigenvalue + largest_eigenvalue) / 2

# Apply Shifted Inverse Power Method
eigenvalue = shifted_inverse_power_method.shifted_inverse_power_method(A, n, shift, tol, max_iter)
print("Eigenvalue near the midpoint:", eigenvalue)