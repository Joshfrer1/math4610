# Math 4610 Fundamentals of Computational Mathematics Software Manual Template File
This is a template file for building an entry in the student software manual project. You should use the formatting below to
define an entry in your software manual.

<hr>

**Author:** Joshua Frerichs

**Language:** Python 3, using python3 interpreter

For example,

    python3 <file_name.py>

**Routine Name:**           back_substitution

**Description/Purpose:** 
Solves a linear system of equations Ux=b where U is an upper triangular matrix and x is a matrix coefficient

**Input:** 
matrix - the upper triangular matrix
x - the matrix coefficient

**Output:** 
b - a solution matrix

**Usage/Example:**

b = back_substitution.back_substitution(U, x)

**Implementation/Code:** 

def back_substitution(matrix, augment):
    """
    Solves the system Ux = b, where U is an upper triangular matrix.
    :param matrix: A 2D list of floats (upper triangular matrix).
    :param augment: a 1D list of floats (augmented to the matrix)
    :return: A 1D list of floats (solution vector of the linear system).
    """
    n = len(matrix)
    solution_vector = [0 for _ in range(n)]  # Initialize the solution vector with zeros

    for i in range(n - 1, -1, -1):  # Start from the last row and go upwards
        if matrix[i][i] == 0:
            raise ValueError("The matrix is singular.")
        
        # Start with the known value from b
        solution_vector[i] = augment[i]
        
        # Subtract the known values of the solved variables
        for j in range(i + 1, n):
            solution_vector[i] -= matrix[i][j] * solution_vector[j]
        
        # Divide by the coefficient of the variable to solve for the system
        solution_vector[i] = solution_vector[i] / matrix[i][i]
    
    return solution_vector

**Routine Name:**           forward_elimination

**Description/Purpose:** 
solve a linear system of equations in lower triangular form

**Input:** 
A - a reduced matrix
b - the reduced augmented matrix

**Output:** 
A matrix reduced to lower triangular

**Usage/Example:**
the use case is for LU decomposition. Refer to that routine to see a complete example of usage

**Implementation/Code:** 
def forward_substitution(L, b):
    """
    Solve a lower triangular matrix using forward substitution method
    :param L: a 2D list of floats (reduced matrix)
    :param b: a 1D list of floats (augment to the reduced matrix)
    :return: a 1D list of floats (solutions to the system)
    """
    # number of equations
    n = len(b)
    # intialize a matrix of zeros for solution matrix
    y = [0 for _ in range(n)]
    # iterate over all rows of L and b
    for i in range(n):
        # accumulate sum of y
        sum = 0
        # iterate over columns of L
        for j in range(i):
            # update sum with product of L_ij and the previously
            # calculated value of y_j
            sum += L[i][j] * y[j]
        # once sum is calculated, subtract sum from y_i. Since L is
        # in lower triangular, the value of y_i can be determined directly
        # wihout needing to divide by the diagonal element L_ij as it is
        # implicity 1.
        y[i] = b[i] - sum
    return y

**Routine Name:**           gaussian_elimination

**Description/Purpose:** 
Just a function to call helpers

**Input:** 
A - matrix
b - coefficient matrix

**Output:** 
returns back_substituion call which returns a solution matrix

**Usage/Example:**
gaussian_elimination.gaussian_elimination(A, b)

**Implementation/Code:** 

def gaussian_elimination(A, b):
    """
    The Gaussian Elimination method using forward elimination to reduce a matrix
    to lower triangular form than using back substitution to solve the matrix
    :param A: a 2D list of floats (matrix)
    :param b: a 1D list of floats (the augment to the matrix)
    :return: a 1D list of floats (solutions to the system)
    """
    # set a tuple to call forward_elimination
    A, b = forward_elimination.forward_elimination(A, b)
    # return back substiution to get solutions
    return back_substitution.back_substitution(A, b)

**Routine Name:**           lu_factorization.lu_decomposition

**Description/Purpose:** 
Reduce a matrix to lower triangular form

**Input:** 
A - a matrix to be reduced

**Output:** 
A reduced matrix

**Usage/Example:**
L, U = lu_factorization.lu_decomposition(A)

**Implementation/Code:** 
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

**Routine Name:**           lu_decomposition.forward_substitution

**Description/Purpose:** 
Solve a lower triangular matrix

**Input:** 
L - a reduced matrix
b - a reduced augment to the matrix

**Output:** 
a matrix of solutions

**Usage/Example:**
b = lu_factorization.forward_substitution(L, flat_list)

**Implementation/Code:**
def forward_substitution(L, b):
    """
    Solve a lower triangular matrix using forward substitution method
    :param L: a 2D list of floats (reduced matrix)
    :param b: a 1D list of floats (augment to the reduced matrix)
    :return: a 1D list of floats (solutions to the system)
    """
    # number of solutions
    n = len(b)
    # set solution matrix to zeros
    y = [0 for _ in range(n)]
    Iterate through L rows
    for i in range(n):
        sum = 0
        Iterate through L columns
        for j in range(i):
            # perform back substitution operations
            sum += L[i][j] * y[j]
        y[i] = b[i] - sum
    return y

**Routine Name:**           matrix_multiplier

**Description/Purpose:** 
Multiplies two matrices that have compatible sizes where rows of A are the same
as columns of B. Done by dot product.

**Input:** 
A - matrix A
B - matrix B

**Output:** 
A product of the matrices.

**Usage/Example:**
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

**Implementation/Code:** 
def matrix_multiplier(A, B):
    """
    Uses the dot product to multiply two matrices
    :param A: A 2D list of floats (matrix A).
    :param B: a 2d list of floats (matrix B)
    :return: the product
    """
    # check that rows of A are same number as columns of B
    if (len(A[0]) != len(B)):
        print("Error, matrices cannot Be multiplied. Matrices are not valid sizes")
        return 0

    # this is fancy list comprehension I learned about researching Pythonic ways
    # to multiply matrices. I know we developed a different way in class but
    # this was written originally as one line of code that I expanded out
    # to explain better. We start from the bottom up
    return [
        [
            # do the dot product over the rows of A and columns of B. Inside
            # the sum function is an iterable which is an inner generator expression
            sum(
                # inner generator expression that pairs elements to then multiply
                A * B 
                for A, B in zip(row_A, col_B)
            )
            # zip(*B) constructs a transpose of B turning its columns
            # into rows to then allow for an iteration over the columns 
            for col_B in zip(*B)
        ]
        # Iterate over rows of matrix A 
        for row_A in A
    ]

**Last Modified:** September/2017