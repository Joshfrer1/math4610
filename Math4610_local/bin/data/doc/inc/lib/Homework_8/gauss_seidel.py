import numpy as np

def gauss_seidel(matrix, b, tolerance=1e-10, max_iterations=1000):
    n = len(matrix)
    x = np.zeros(n)
    operation_count = 0

    for _ in range(max_iterations):
        x_new = x.copy()
        operation_count += n  # Count the operations for copying the array

        for i in range(n):
            s1 = sum(matrix[i][j] * x_new[j] for j in range(i))
            s2 = sum(matrix[i][j] * x[j] for j in range(i + 1, n))
            operation_count += 3 * i + 3 * (n - i - 1)  # Count multiplications and additions

            if matrix[i][i] != 0:
                x_new[i] = (b[i] - s1 - s2) / matrix[i][i]
                operation_count += 3  # One subtraction and one division
            else:
                x_new[i] = 0

        if np.allclose(x, x_new, atol=tolerance):
            return x_new, operation_count

        x = x_new

    return x, operation_count
