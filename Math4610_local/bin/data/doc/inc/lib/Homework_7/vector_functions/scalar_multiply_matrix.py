def scalar_multiply_matrix(scalar, matrix):
    return [[scalar * matrix[i][j] for j in range(len(matrix[0]))] for i in range(len(matrix))]