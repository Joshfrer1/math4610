from vector_functions import power_method, matrix_addition, scalar_multiply_matrix, identity_matrix
import inverse_power_method


def shifted_inverse_power_method(matrix, guess, shift, tol, max_iter):
    shifted_matrix = matrix_addition.matrix_addition(matrix, scalar_multiply_matrix.scalar_multiply_matrix(-shift, identity_matrix.identity_matrix(len(matrix))))
    return inverse_power_method.inverse_power_method(shifted_matrix, guess, tol, max_iter)

