from vector_functions import vector_subtract, normalize_vector, vector_norm

def leslie_matrix_vector_multiply(leslie_matrix, vector):
    result = [0] * len(vector)
    # First row of Leslie matrix
    result[0] = sum(x * y for x, y in zip(leslie_matrix[0], vector))
    # Sub-diagonal elements
    for i in range(1, len(vector)):
        result[i] = leslie_matrix[i][i - 1] * vector[i - 1]
    return result

def leslie_power_method(leslie_matrix, n, tol, max_iter):
    b = normalize_vector.normalize_vector(n)
    for _ in range(max_iter):
        n = leslie_matrix_vector_multiply(leslie_matrix, b)
        b_next = normalize_vector.normalize_vector(n)
        if vector_norm.vector_norm(vector_subtract.vector_subtract(b_next, b)) < tol:
            break
        b = b_next
    max_value = sum(leslie_matrix_vector_multiply(leslie_matrix, b)[i] * b[i] for i in range(len(b))) / sum(b[i] * b[i] for i in range(len(b)))
    return max_value

# Example usage
leslie_matrix = [[0.1, 2.0, 1.5, 0.5], [0.5, 0, 0, 0], [0, 0.7, 0, 0], [0, 0, 0.9, 0]]
n = [1, 1, 1, 1]
tol = 1e-6
max_iter = 1000


