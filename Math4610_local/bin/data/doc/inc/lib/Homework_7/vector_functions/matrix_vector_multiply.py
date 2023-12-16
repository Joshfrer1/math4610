def matrix_vector_multiply(matrix, vector):
    return [sum(x * y for x, y in zip(row, vector)) for row in matrix]