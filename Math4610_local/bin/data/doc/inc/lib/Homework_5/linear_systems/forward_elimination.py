def forward_elimination(A, b):
    n = len(A)
    for i in range(n):
        # Search for maximum in this column
        max_el = abs(A[i][i])
        max_row = i
        for k in range(i+1, n):
            if abs(A[k][i]) > max_el:
                max_el = abs(A[k][i])
                max_row = k

        # Swap maximum row with current row (column by column)
        for k in range(i, n):
            A[max_row][k], A[i][k] = A[i][k], A[max_row][k]
        # Swap b value
        b[max_row], b[i] = b[i], b[max_row]

        # Make all rows below this one 0 in current column
        for k in range(i+1, n):
            c = -A[k][i] / A[i][i]
            for j in range(i, n):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]
            # Subtract the same factor from the b vector as well
            b[k] += c * b[i]

    return A, b
