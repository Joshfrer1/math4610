import random

def create_diagonally_dominant_matrix(n):
    """Create a diagonally dominant matrix of size n x n."""
    matrix = [[random.random() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        sum_of_row = sum(abs(matrix[i][j]) for j in range(n) if i != j)
        matrix[i][i] = sum_of_row + random.random() * 10
    return matrix