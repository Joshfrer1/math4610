def matrix_multiplier(A, B):
    """
    Uses the dot product to multiply two matrices
    :param A: A 2D list of floats (matrix A).
    :param B: a 2d list of floats (matrix B)
    :return: the product
    """
    if (len(A[0]) != len(B)):
        print("Error, matrices cannot Be multiplied. Matrices are not valid sizes")
        return 0
    return [
        [
            sum(
                A * B 
                for A, B in zip(row_A, col_B)
            ) 
            for col_B in zip(*B)
        ] 
        for row_A in A
    ]


