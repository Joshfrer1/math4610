from vector_functions import normalize_vector, vector_norm, gaussian_elimination, matrix_vector_multiply

def inverse_power_method(matrix, guess, tol, max_iter):
    n = len(matrix)
    b = normalize_vector.normalize_vector(guess)
    for _ in range(max_iter):
        # Augmenting matrix with b and applying Gaussian elimination
        augmented_matrix = [row[:] + [b[i]] for i, row in enumerate(matrix)]
        y = gaussian_elimination.gaussian_elimination(augmented_matrix)
        b_next = normalize_vector.normalize_vector(y)
        if vector_norm.vector_norm([b_next[i] - b[i] for i in range(n)]) < tol:
            break
        b = b_next

    # Approximating the eigenvalue
    mb = matrix_vector_multiply.matrix_vector_multiply(matrix, b)
    eigenvalue = sum(mb[i] * b[i] for i in range(n)) / sum(b[i] * b[i] for i in range(n))
    return eigenvalue

