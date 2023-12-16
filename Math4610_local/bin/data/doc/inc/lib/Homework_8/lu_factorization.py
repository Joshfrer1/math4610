def lu_decomposition(matrix):
    n = len(matrix)
    operation_count = 0

    # Initialize L and U matrices
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
                operation_count += 2  # One multiplication, one addition
            U[i][k] = matrix[i][k] - sum
            operation_count += 1  # One subtraction

        # Lower Triangular
        for k in range(i, n):
            if i == k:
                L[i][i] = 1  # Diagonal as 1
            else:
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])
                    operation_count += 2  # One multiplication, one addition
                L[k][i] = (matrix[k][i] - sum) / U[i][i]
                operation_count += 2  # One subtraction, one division

    return L, U, operation_count

def forward_substitution(L, b):
    n = len(L)
    y = [0] * n
    operation_count = 0

    for i in range(n):
        sum = 0
        for j in range(i):
            sum += L[i][j] * y[j]
            operation_count += 2  # One multiplication, one addition
        y[i] = b[i] - sum
        operation_count += 1  # One subtraction

    return y, operation_count

def backward_substitution(U, y):
    n = len(U)
    x = [0] * n
    operation_count = 0

    for i in range(n - 1, -1, -1):
        sum = 0
        for j in range(i + 1, n):
            sum += U[i][j] * x[j]
            operation_count += 2  # One multiplication, one addition
        x[i] = (y[i] - sum) / U[i][i]
        operation_count += 2  # One subtraction, one division

    return x, operation_count