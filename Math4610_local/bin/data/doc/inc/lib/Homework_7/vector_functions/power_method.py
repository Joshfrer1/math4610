from vector_functions import normalize_vector, matrix_vector_multiply, vector_norm

def power_method(matrix, guess, tol, max_iter):
    b = normalize_vector.normalize_vector(guess)
    for _ in range(max_iter):
        n = matrix_vector_multiply.matrix_vector_multiply(matrix, b)
        b_next = normalize_vector.normalize_vector(n)
        if vector_norm.vector_norm([b_next[i] - b[i] for i in range(len(matrix))]) < tol:
            break
        b = b_next
    eigenvalue = sum(matrix_vector_multiply.matrix_vector_multiply(matrix, b)[i] * b[i] for i in range(len(matrix))) / sum(b[i] * b[i] for i in range(len(matrix)))
    return eigenvalue
