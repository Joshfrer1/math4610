from linear_systems import *

def test_upper_triangular():
    U = [
    [3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 5, 1, 3, 4, 5, 6, 7, 8, 9],
    [0, 0, 6, 1, 4, 5, 6, 7, 8, 9],
    [0, 0, 0, 7, 4, 5, 6, 7, 8, 9],
    [0, 0, 0, 0, 9, 5, 6, 7, 8, 9],
    [0, 0, 0, 0, 0, 8, 6, 7, 8, 9],
    [0, 0, 0, 0, 0, 0, 5, 7, 8, 9],
    [0, 0, 0, 0, 0, 0, 0, 9, 8, 9],
    [0, 0, 0, 0, 0, 0, 0, 0, 5, 9],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
    ]

    b = [5, 1, 0, 8, 10, 3, 2, 1, 9, 5]

    solution = back_substitution.back_substitution(U, b)
    print("Upper Triangular Solution:\n", solution)
    
def test_lower_triangular():
    A = [
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
    [1, 2, 3, 4, 5, 1, 0, 0, 0, 0],
    [1, 2, 3, 4, 5, 2, 3, 0, 0, 0],
    [1, 2, 3, 4, 5, 1, 1, 1, 0, 0],
    [1, 2, 3, 4, 5, 1, 1, 1, 5, 0],
    [1, 2, 3, 4, 5, 1, 1, 1, 1, 4],
    ]

    b = [4, 2, 4, 2, 1, 4, 2, 1, 1, 1]

    # Perform LU decomposition
    L, U = lu_factorization.lu_decomposition(A)

    # Solve Ly = b for y
    y = lu_factorization.forward_substitution(L, b)

    # Solve Ux = y for x
    x = back_substitution.back_substitution(U, y)

    print("Lower Triangular Solution:\n", x)
    
def test_matrix_multiplier():
    
    A = [
        [3, 1, 1],
        [2, 6, 3],
        [1, 1, 5]
    ]
    
    B = [
        [1],
        [1],
        [1]
    ]
    
    result = matrix_multiplier.matrix_multiplier(A, B)
    print("Multiplied Matrix:\n")
    for row in result:
        print(row)
    

def test_gaussian():
    A = [
        [3, 1, 1],
        [2, 6, 3],
        [1, 1, 5]
    ]
    
    y = [
        [1],
        [1],
        [1]
    ]   
    
    b = matrix_multiplier.matrix_multiplier(A, y)
    flat_list = [num for sublist in b for num in sublist]
    x = gaussian_elimination.gaussian_elimination(A, flat_list)
    print("Gaussian Elimination x:\n", x)
    
def test_lu_factorization():
    # Example usage:
    A = [
        [3, 0, 0, 0],
        [2, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 1, 1, 1]
    ]

    y = [
        [1],
        [1],
        [1],
        [1]
    ] 
    
    z = matrix_multiplier.matrix_multiplier(A, y)
    flat_list = [num for sublist in z for num in sublist]


    # Perform LU decomposition
    L, U = lu_factorization.lu_decomposition(A)

    # Solve Ly = b for y
    b = lu_factorization.forward_substitution(L, flat_list)

    # Solve Ux = y for x
    x = back_substitution.back_substitution(U, b)

    print("Solution vector x:\n", x)
        
test_upper_triangular()
test_lower_triangular()
test_gaussian()
test_lu_factorization()