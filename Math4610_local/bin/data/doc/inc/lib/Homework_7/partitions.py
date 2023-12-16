from vector_functions import matrix_addition, scalar_multiply_matrix, identity_matrix, normalize_vector, matrix_vector_multiply, vector_norm, power_method
import inverse_power_method
def power_method_shifted(matrix, guess, shift, tol, max_iter):
    shifted_matrix = matrix_addition.matrix_addition(matrix, scalar_multiply_matrix.scalar_multiply_matrix(-shift, identity_matrix.identity_matrix(len(matrix))))
    b = normalize_vector.normalize_vector(guess)
    for _ in range(max_iter):
        n = matrix_vector_multiply.matrix_vector_multiply(shifted_matrix, b)
        b_next = normalize_vector.normalize_vector(n)
        if vector_norm.vector_norm([b_next[i] - b[i] for i in range(len(matrix))]) < tol:
            break
        b = b_next
    eigenvalue = sum(matrix_vector_multiply.matrix_vector_multiply(shifted_matrix, b)[i] * b[i] for i in range(len(matrix))) / sum(b[i] * b[i] for i in range(len(matrix)))
    return eigenvalue + shift


