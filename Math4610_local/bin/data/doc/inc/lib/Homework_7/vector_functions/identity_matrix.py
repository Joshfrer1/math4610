def identity_matrix(size):
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]