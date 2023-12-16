import numpy as np

def jacobi_iteration(matrix, b, tolerance=1e-10, max_iterations=1000):
    n = len(matrix)
    x = np.zeros(n)
    x_new = np.zeros(n)
    operation_count = 0

    for _ in range(max_iterations):
        for i in range(n):
            s = 0
            for j in range(n):
                if i != j:
                    s += matrix[i][j] * x[j]
                    operation_count += 2  # One multiplication, one addition

            if matrix[i][i] != 0:
                x_new[i] = (b[i] - s) / matrix[i][i]
                operation_count += 2  # One subtraction, one division
            else:
                x_new[i] = 0

        if np.allclose(x, x_new, atol=tolerance):
            return x_new, operation_count

        x = np.copy(x_new)
        operation_count += n  # Count the operations for copying the array

    return x, operation_count

