def lu_decomposition(A):
    """
    Reduce a matrix to a lower triangular matrix using LU decomposition
    :param A: a 2D list (matrix)
    :return: a tuple
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
            U[i][k] = A[i][k] - sum

        # Lower Triangular
        for k in range(i, n):
            if i == k:
                L[i][i] = 1  # Diagonal as 1
            else:
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])
                L[k][i] = (A[k][i] - sum) / U[i][i]

    return L, U

def forward_substitution(L, b):
    """
    Solve a lower triangular matrix using forward substitution method
    :param L: a 2D list of floats (reduced matrix)
    :param b: a 1D list of floats (augment to the reduced matrix)
    :return: a 1D list of floats (solutions to the system)
    """
    n = len(b)
    y = [0 for _ in range(n)]
    for i in range(n):
        sum = 0
        for j in range(i):
            sum += L[i][j] * y[j]
        y[i] = b[i] - sum
    return y

