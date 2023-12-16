# Math 4610 Fundamentals of Computational Mathematics Software Manual
Welcome to Joshua Frerichs' software manual. This manual is designed to contain all the methods used in Math 4610 and their explainations of usage. It is organized by a few sections. 

<hr>

**Author:** Joshua Frerichs

**Language:** Python 3, using python3 interpreter

[Approximations and Tolerances](#approximations-and-tolerances)
 - [Machine Epsilon of 64bit Floats](#machine-epsilon-of-64bit-floats)
 - [Machine Epsilon of 32bit Floats](#machine-epsilon-of-32bit-floats)

[Matrix and Vector Operations](#matrix-and-vector-operations)
 - [L1 Distance](#l1-distance)
 - [L1 Norm](#l1-norm)
 - [L2 Distance](#l2-distance)
 - [L2 Norm](#l2-norm)
 - [LInfinity Distance](#linfinity-distance)
 - [LInfinity Norm](#linfinity-norm)
 - [Identity Matrix](#identity-matrix)
 - [Matrix Addition](#matrix-addition)
 - [Scalar Multiplication](#scalar-multiplication)
 - [Matrix Multiplication](#matrix-multiplication)
 - [Vector Subtraction](#vector-subtraction)
 - [Normalizing a Vector](#normalizing-a-vector)

[Methods for Solving Linear Systems of Equations](#methods-for-solving-linear-systems-of-equations)
 - [Back Substituion](#back-substitution)
 - [Forward Substitution](#forward-substitution)
 - [Forward Elimination](#forward-elimination)
 - [LU Factorization](#lu-factorization)
 - [Gaussian Elimination](#gaussian-elimination)

[Eigenvalue Finding Methods](#eigenvalue-finding-methods)
 - [The Power Method](#the-power-method)

[Root Finding Methods](#root-finding-methods)
 - [Newtons Method](#newtons-method)
 - [The Secant Method](#the-secant-method)
 - [The Bisection Method](#the-bisection-method)
 - [A Hybrid Method](#a-hybrid-method)
 - [Inverse Power Method](#inverse-power-method)
 - [Shifted Inverse Power Method](#shifted-inverse-power-method)

[Optimizing Methods](#optimizing-methods)
 - [Jacobi Iteration](#jacobi-iteration)
 - [Gauss-Seidel Iteration](#gauss-seidel-iteration)
 - [Comparing Jacobi, Gauss-Seidel, and LU Factorization](#comparing-jacobi-gauss-seidel-and-lu-factorization)

[Leslie Matrices](#leslie-matrices)
 - [Leslie Matrices and the Power Method](#leslie-matrices-and-the-power-method)
 - [Leslie Matrices and Jacobian Iteration](#leslie-matrices-and-jacobian-iteration)

[openMP](#openmp)
 - [Jacobi Iteration Optimized](#jacobi-iteration-optimized)
 - [The Power Method Optimzied](#the-power-method-optimized)

# Approximations and Tolerances
## Machine Epsilon of 64bit Floats
**Routine Name:**           dmaceps

For example,

    gcc dmaceps.c -o dmaceps

will produce a .o file such that you can run:

    ./dmaceps

**Description/Purpose:** This function will compute the double precision value for the machine epsilon or the number of digits
in the representation of real numbers in double precision. This is a function for analyzing the behavior of any computer. This usually will need to be run one time for each computer.

**Input:** There are no inputs needed in this case.

**Output:** This function is of type void. We will simply display the output.

**Usage/Example:**

The function does not take any parameters. We can call the function inside the main function by:

      machineEps32()

Output from the lines above:

      2.220446e-16

The first value (24) is the number of binary digits that define the machine epsilon and the second is related to the decimal version of the same value. The number of decimal digits that can be represented is roughly eight (E-08 on the end of the second value).

**Implementation/Code:** The following is the code for smaceps()

    #include <stdio.h>
    #include <math.h>
    #include <stdlib.h>

We include the following libraries to have access to standard input/output functions, the math library to specify our double type.

    void machineEps64() {
        double previous_eps = 0.0;
        double eps = 1.0;

We initialize two double values to compare. You could pick any number with a difference of 1. For our use case and ease, we use 0.0 and 1.0 respectively.

        while ((1.0 + eps) != 1.0){
            previous_eps = eps;
            eps /= 2.0;
        }
While our epsilon plus 1 is not equal to one, we will set our previous_eps to eps and then divide equals our epp by 2.0. the /= is equivalent to

    eps = eps / 2.0


What is happening is that we are summating each iteration of our epsilon to compare to some value really close to 1.0. We keep diving our eps by 2.0 and setting it as our new eps to compare in the while loop until we reach machine precision for a double.

The we print out our result with a simple print statement:

        printf("%e\n", previous_eps);
    }


## Machine Epsilon of 32bit Floats
**Routine Name:**           smaceps

For example,

    gcc smaceps.c -o smaceps

will produce a .o file such that you can run:

    ./smaceps

**Description/Purpose:** This function will compute the single precision value for the machine epsilon or the number of digits in the representation of real numbers in single precision. This is a function for analyzing the behavior of any computer. This usually will need to be run one time for each computer.

**Input:** There are no inputs needed in this case. The function is of type void.

**Output:** This function will output the value of single precision.

**Usage/Example:**

The function does not take any arguements and is simply called in a main function by:

      machineEps32()

Output from the call above:

      1.192093e-07

The first value (24) is the number of binary digits that define the machine epsilon and the second is related to the
decimal version of the same value. The number of decimal digits that can be represented is roughly eight (E-07 on the
end of the second value).

**Implementation/Code:** The following is the code for smaceps()

    #include <stdio.h>
    #include <math.h>
    #include <stdlib.h>

We include the following libraries to have access to standard input/output functions, the math library to specify our float type.

    void machineEps32() {
        float previous_eps = 0.0f;
        float eps = 1.0f;

We initialize two floating point values to compare. You could pick any number with a difference of 1. For our use case and ease, we use 0.0 and 1.0 respectively.

        while ((1.0f + eps) != 1.0f){
            previous_eps = eps;
            eps /= 2.0f;
        }
While our epsilon plus 1 is not equal to one, we will set our previous_eps to eps and then divide equals our epp by 2.0f. the /= is equivalent to

    eps = eps / 2.0f


What is happening is that we are summating each iteration of our epsilon to compare to some value really close to 1.0f. We keep diving our eps by 2.0 and setting it as our new eps to compare in the while loop until we reach machine precision for a float.

The we print out our result with a simple print statement:

        printf("%e\n", previous_eps);
    }

# Matrix and Vector Operations
This section contains various matrix and vector operations used in class and amongst some functions in later practices.

## L1 Distance

**Routine Name:** l1_distance

**Description/Purpose:**
Compute the distance of two vectors in L1 space

**Input:** 
two vectors that must be the same size per the zip() function

**Output:**
the distance between two vectors

**Usage/Example:**
```python
print(l1_distance([1,2], [3,4]))
```

**Implementation/Code:**
```python
def l1_distance(u, v):
    distance = 0
    for i, j in zip(u, v):
        distance += abs(i - j)  
    return distance
``` 

## L1 Norm

**Routine Name:** l_1_norm

**Description/Purpose:** 
computes the normal of a vector in L1 space

**Input:** 
a vector

**Output:**
the norm of that vector

**Usage/Example:**
```python
print(l_1_norm([1,2,3]))
```

**Implementation/Code:**
```python
def l_1_norm(vector):
    length = 0
    for i in vector:
        length += abs(i)
    return length
```  

## L2 Distance

**Routine Name:** l2_distance

**Description/Purpose:** 
gives a distance between two vectors in the L2 space

**Input:** 
two vectors of the same size

**Output:**
the distance between those two vectors

**Usage/Example:**
```python
print(l2_distance([1,5,3], [3,2,1]))
```

**Implementation/Code:**
```python
def l2_distance(u, v):
    distance = 0
    for i, j in zip(u, v):
        distance += (i - j)**2
        
    return distance
```  

## L2 Norm

**Routine Name:** l_2_norm

**Description/Purpose:** 
returns the normal of a vector in L2 space

**Input:** 
a vector

**Output:**
the normal of that vector

**Usage/Example:**
```python
print(l_2_nom([1,2,3]))
```

**Implementation/Code:**
```python
def l_2_norm(vector):
    length = 0
    for i in vector:
        length += i**2
    return m.sqrt(length)
``` 

## LInfinity Distance

**Routine Name:** infinity_distance

**Description/Purpose:** 
gets the distance between two vectors in infinity space

**Input:** 
two vectors

**Output:**
the distance between those two vectors

**Usage/Example:**
```python
print(infinity_distance([1,2,3], [100, 1000, 10000]))
```

**Implementation/Code:**
```python
def infinity_distance(u, v):
    return abs(infinity_norm(u) - infinity_norm(v))
``` 

## LInfinity Norm

**Routine Name:** infinity_norm

**Description/Purpose:** 
gets the normal of a vector in infinity space

**Input:** 
a vector

**Output:**
the normal of that vector

**Usage/Example:**
```python
print(infinity_norm([1,100,11]))
```

**Implementation/Code:**
```python
def infinity_norm(vector):
    max = vector[0]
    for i in vector:
        abs_i = abs(i)
        if max < abs_i:
            max = abs_i
    return max
``` 

## Identity Matrix

**Routine Name:** identity_matrix

**Description/Purpose:** 
returns the identity matrix of size N

**Input:** 
the size of the NxN matrix

**Output:**
the identity of that matrix

**Usage/Example:**
was used in the shifted_inverse_power_method

**Implementation/Code:**
```python
def identity_matrix(size):
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]
``` 

## Matrix Addition

**Routine Name:** matrix_addition

**Description/Purpose:** 
Gets the sum of two matrices and returns their sums in a new matrix

**Input:** 
two NxN matrices

**Output:**
an NxN matrix with their sums

**Usage/Example:**
used all over in root finding methods section

**Implementation/Code:**
```python
def matrix_addition(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
``` 

## Scalar Multiplication

**Routine Name:** scalar_matrix_multiply

**Description/Purpose:** 
multiplies a matrix by a scalar

**Input:** 
a matrix and a scalar

**Output:**
the matrix multiplied by the scalar

**Usage/Example:**
used in root finding methods section

**Implementation/Code:**
```python
def scalar_multiply_matrix(scalar, matrix):
    return [[scalar * matrix[i][j] for j in range(len(matrix[0]))] for i in range(len(matrix))]
``` 

## Matrix Multiplication

**Routine Name:** matrix_vector_multiply

**Description/Purpose:** 
does the dot product on two matrices

**Input:** 
a matrix and a vector

**Output:**
their newly multiplied matrix

**Usage/Example:**
used in root finding section

**Implementation/Code**
```python
def matrix_vector_multiply(matrix, vector):
    return [sum(x * y for x, y in zip(row, vector)) for row in matrix]
```

## Vector Subtraction

**Routine Name:**  vector_subtraction

**Description/Purpose:** 
subtracts two vectors

**Input:** 
two vectors

**Output:**
a vector resulting from the two vectors subtracting

**Usage/Example:**
used in root finding section

**Implementation/Code:**
```python
def vector_subtract(v1, v2):
    return [x - y for x, y in zip(v1, v2)]
``` 

## Normalizing a Vector

**Routine Name:** normalize_vector, vector_norm

**Description/Purpose:** 
gets the norm of a vector and divides it by a maginitude to get a normalized vector

**Input:** 
a vector to get the norm and magnitude

**Output:**
a normalized vector

**Usage/Example:**
used in root finding section

**Implementation/Code:**
```python
def vector_norm(vector):
    return sum(x**2 for x in vector) ** 0.5

def normalize_vector(vector):
    norm = vector_norm.vector_norm(vector)
    return [x / norm for x in vector]
``` 

# Methods for Solving Linear Systems of Equations

## Back Substitution

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
```python
def back_substitution(matrix, augment):
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
```

## Forward Substitution
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
```python
def forward_substitution(L, b):
    # number of solutions
    n = len(b)
    # set solution matrix to zeros
    y = [0 for _ in range(n)]
    # Iterate through L rows
    for i in range(n):
        sum = 0
        Iterate through L columns
        for j in range(i):
            # perform back substitution operations
            sum += L[i][j] * y[j]
        y[i] = b[i] - sum
    return y
```

## Forward Elimination

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
```python
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
```

## LU Factorization

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
```python
def lu_decomposition(A):
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
``` 

## Gaussian Elimination

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

```python
def gaussian_elimination(A, b):
    A, b = forward_elimination.forward_elimination(A, b)
    # return back substiution to get solutions
    return back_substitution.back_substitution(A, b)
```


# Eigenvalue Finding Methods

## The Power Method

**Routine Name:** power_method

**Description/Purpose:** 
the power method is a computational way to approximate the largest eigenvalue of a matrix

**Input:** 
a matrix, an intial guess, a tolerance for approximation, and a max number of iterations

**Output:**
an approximated eigenvalue

**Usage/Example:**
```python
A = [[4, 2], [1, 3]]
n = [1, 0]
tol = 1e-6
max_iter = 1000

max_value = power_method(A, n, tol, max_iter)
print("Largest Eigenvalue:", max_value)
```

**Implementation/Code:**
```python
def power_method(matrix, guess, tol, max_iter):
    b = normalize_vector.normalize_vector(guess)
    for _ in range(max_iter):
        n = matrix_vector_multiply.matrix_vector_multiply(matrix, b)
        b_next = normalize_vector.normalize_vector(n)
        if vector_norm.vector_norm([b_next[i] - b[i] for i in range(len(matrix))]) < tol:
            break
        b = b_next
    eigenvalue = sum(matrix_vector_multiply.matrix_vector_multiply(matrix, b)[i] * b[i] for i in range(len(matrix))) / sum(b[i] * b[i] for i in range(len(matrix)))
    return eigenvalue
```

## Inverse Power Method

**Routine Name:** inverse_power_method

**Description/Purpose:** 
Used to approximate the smallest eigenvalue of a matrix

**Input:** 
a matrix, an initial guess, a tolerace for approximation, and a max number of iterations

**Output:**
an approximation of the smallest eigenvalue of a matrix

**Usage/Example:**
```python
A = [[4, 2], [1, 3]]
n = [1, 0]
tol = 1e-6
max_iter = 1000
smallest_eigenvalue = inverse_power_method.inverse_power_method(A, n, tol, max_iter)
print("Smallest Eigenvalue:", smallest_eigenvalue)
```

**Implementation/Code:**
```python
def inverse_power_method(matrix, guess, tol, max_iter):
    n = len(matrix)
    b = normalize_vector.normalize_vector(guess)
    for _ in range(max_iter):
        # Augmenting matrix with b and applying Gaussian elimination
        augmented_matrix = [row[:] + [b[i]] for i, row in enumerate(matrix)]
        y = gaussian_elimination.gaussian_elimination(augmented_matrix)
        b_next = normalize_vector.normalize_vector(y)
        if vector_norm.vector_norm([b_next[i] - b[i] for i in range(n)]) < tol:
            break
        b = b_next

    # Approximating the eigenvalue
    mb = matrix_vector_multiply.matrix_vector_multiply(matrix, b)
    eigenvalue = sum(mb[i] * b[i] for i in range(n)) / sum(b[i] * b[i] for i in range(n))
    return eigenvalue
```

## Shifted Inverse Power Method

**Routine Name:** shifted_inverse_power_method

**Description/Purpose:** 
allows a user to target specific eigenvalues that don't have to be the largest in magnitude. For the homework I was to find the midpoint between the smallest and largest eigenvalue

**Input:** 
a matrix, an initial guess, a shift, a tolerace for approximation, and a max number of iterations

**Output:**
the targeted eigenvalue

**Usage/Example:**
look at homework_7.py in lib/Homework_7, its too much to put here. Look at the partitions example.

**Implementation/Code:**
```python
def shifted_inverse_power_method(matrix, guess, shift, tol, max_iter):
    shifted_matrix = matrix_addition.matrix_addition(matrix, scalar_multiply_matrix.scalar_multiply_matrix(-shift, identity_matrix.identity_matrix(len(matrix))))
    return inverse_power_method.inverse_power_method(shifted_matrix, guess, tol, max_iter)
``` 

# Root Finding Methods

## Newtons Method

**Routine Name:** newtons_method

**Description/Purpose:** 
one of various ways to computationally approximate finding roots of real valued functions. Requires the knowledge of the derivative of the function you are trying to find roots for

**Input:** 
an initial guess, a tolerance for approximation, and a max number of iterations

**Output:**
found roots

**Usage/Example:**
```python
tolerance = 1e-6
max_iterations = 100

# For the root near 2
root1 = newtons_method(2, tolerance, max_iterations)

# For the root near 3
root2 = newtons_method(3, tolerance, max_iterations)

print("Root near 2:", root1)
print("Root near 3:", root2)
```

**Implementation/Code:**
```python
def f(x):
    return x**2 - 5*x + 6

def df(x):
    return 2*x - 5

def newtons_method(initial_guess, tol, max_iter):
    x = initial_guess
    for _ in range(max_iter):
        x_new = x - f(x) / df(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x  # Return the last approximation if tolerance not met
``` 

## The Secant Method

**Routine Name:** secant_method

**Description/Purpose:** 
another useful technique that improves on Newtons method by not requiring the knowledge of a second derivative

**Input:**
two initial guesses for roots and a tolerance for approximation

**Output:**
a tuple that has both approximated roots

**Usage/Example:**
```python
tolerance = 1e-6
max_iterations = 100

# For the root near 2
root1 = secant_method(1.5, 2.5, tolerance, max_iterations)

# For the root near 3
root2 = secant_method(2.5, 3.5, tolerance, max_iterations)

print("Root near 2:", root1)
print("Root near 3:", root2)
```

**Implementation/Code:**
```python
def secant_method(x0, x1, tol):
    for _ in range(100):  # Assuming a max of 100 iterations
        if abs(f(x1) - f(x0)) < tol:
            raise ValueError('Division by zero in secant method')
        x_new = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x_new - x1) < tol:
            return x_new
        x0, x1 = x1, x_new
    return x1
``` 

## The Bisection Method

**Routine Name:** bisection_method

**Description/Purpose:** 
useful for finding roots of continuous functions with sign changes also known as the binary search

**Input:** 
two intervals where a root is believed to be found and a tolerance for approximation

**Output:**
the approximated root being looked for

**Usage/Example:**
```python
intervals = [(1, 2.5), (2.5, 4)]  # Adjusted intervals
tolerance = 0.001

roots = []
for interval in intervals:
    root = bisection_method(interval[0], interval[1], tolerance)
    if root is not None:
        roots.append(root)

print("Roots:", roots)
```

**Implementation/Code:**
```python
def bisection_method(a, b, tol):
    if f(a) * f(b) >= 0:
        print("No root found in the interval [", a, ",", b, "]")
        return None

    while (b - a) / 2.0 > tol:
        midpoint = (a + b) / 2.0
        if f(midpoint) == 0:
            return midpoint
        elif f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
    return (a + b) / 2.0
```  

## A Hybrid Method

**Routine Name:** hybrid_method with helper function bisection_for_interval

**Description/Purpose:** 
uses the bisection method to narrow down the location of a root and the secant method to speed up finding that root

**Input:** 
for the bisection_for_interval you need two values that make up an interval and a tolerance for approximation

for the secant method written a little differently, the interval is actually a tuple with the interval for finding the location of roots, and a tolerance for approximation

**Output:**
found roots

**Usage/Example:**
```python
# Example usage
intervals = [(1, 2.5), (2.5, 4)]
tolerance = 0.0001

roots = []
for interval in intervals:
    root = hybrid_method(interval, tolerance)
    if root is not None:
        roots.append(root)

print("Roots:", roots)
```

**Implementation/Code:**
```python
def bisection_for_interval(a, b, tol):
    if f(a) * f(b) >= 0:
        return None, None

    while (b - a) / 2.0 > tol:
        midpoint = (a + b) / 2.0
        if f(midpoint) == 0:
            return midpoint, midpoint
        elif f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint

    return a, b

def hybrid_method(interval, tol):
    a, b = bisection_for_interval(interval[0], interval[1], tol)
    if a is None:
        return None
    if a == b:  # Root found directly by bisection
        return a
    return secant_method.secant_method(a, b, tol)
``` 

# Optimizing Methods

## Jacobi Iteration

**Routine Name:** jacobi_iteration

**Description/Purpose:** 
uses the jacobi method to find solutions to linear systems of equations for diagonlly dominant matricies. tested on a 100x100 matrix

**Input:** 
a diagonally dominant matrix, an augment, a tolerance for approximation, and a max iterations

**Output:**
n solutions where n is the size of the diagonally dominant matrix and a count of operations performed

**Usage/Example:**
```python
# Parameters
n = 100
tolerance = 1e-10
max_iterations = 1000

# Generate a 100 by 100 diagonally dominant matrix and a random b vector
A = diagonal_matrix_generator.create_diagonally_dominant_matrix(n)
b = [random.random() for _ in range(n)]

# Apply the Jacobi iteration method
jacobi_iteration_solution, jacobi_count = jacobi_iteration.jacobi_iteration(A, b, tolerance, max_iterations)
print()
# Print the solution
print("Jacobi Solution:")
print(jacobi_iteration_solution)
```

**Implementation/Code:**
```python
def jacobi_iteration(matrix, b, tolerance=1e-10, max_iterations=1000):
    n = len(matrix)
    x = np.zeros(n)
    x_new = np.zeros(n)
    operation_count = 0

    for _ in range(max_iterations):
        for i in range(n):
            s = 0
            for j in range(n):
                if i != j:
                    s += matrix[i][j] * x[j]
                    operation_count += 2  # One multiplication, one addition

            if matrix[i][i] != 0:
                x_new[i] = (b[i] - s) / matrix[i][i]
                operation_count += 2  # One subtraction, one division
            else:
                x_new[i] = 0

        if np.allclose(x, x_new, atol=tolerance):
            return x_new, operation_count

        x = np.copy(x_new)
        operation_count += n  # Count the operations for copying the array

    return x, operation_count
``` 

## Gauss-Seidel Iteration

**Routine Name:** gauss_seidel

**Description/Purpose:** 
an alternative iteration method to solve linear systems of equations that converges to solutions faster but cannot parallelize as well as the jacobi method.

**Input:** 
a diagonally dominant matrix, a solutions vector, a tolerance for approximation, and a max number of iterations

**Output:**
solutions to the systems of equations and a count of operations performed in a tuple

**Usage/Example:**
```python
gauss_seidel_solution, gs_count = gauss_seidel.gauss_seidel(A, b, tolerance, max_iterations)

# Print the solution
print("Gauss-Seidel Solution:")
print(gauss_seidel_solution)
```
gauss_seidel_solution, gs_count = gauss_seidel.gauss_seidel(A, b, tolerance, max_iterations)

**Implementation/Code:**
```python
import numpy as np

def gauss_seidel(matrix, b, tolerance=1e-10, max_iterations=1000):
    n = len(matrix)
    x = np.zeros(n)
    operation_count = 0

    for _ in range(max_iterations):
        x_new = x.copy()
        operation_count += n  # Count the operations for copying the array

        for i in range(n):
            s1 = sum(matrix[i][j] * x_new[j] for j in range(i))
            s2 = sum(matrix[i][j] * x[j] for j in range(i + 1, n))
            operation_count += 3 * i + 3 * (n - i - 1)  # Count multiplications and additions

            if matrix[i][i] != 0:
                x_new[i] = (b[i] - s1 - s2) / matrix[i][i]
                operation_count += 3  # One subtraction and one division
            else:
                x_new[i] = 0

        if np.allclose(x, x_new, atol=tolerance):
            return x_new, operation_count

        x = x_new

    return x, operation_count
``` 

# Comparing Jacobi, Gauss-Seidel, and LU Factorization

**Routine Name:** lu_decomposition

**Description/Purpose:** 
updated to include a counter and better written lu factorization in order to compare number of operations performed

**Input:** 
a diagonally dominant matrix

**Output:**
solutions and a count of operations

**Usage/Example:**
```python
# Compare LU Factorization, Jacobi, and Gauss-Seidel operations
# LU Factorization
L, U, count_lu = lu_factorization.lu_decomposition(A)

# Solving Ly = b
y, count_forward = lu_factorization.forward_substitution(L, b)

# Solving Ux = y
x, count_backward = lu_factorization.backward_substitution(U, y)

# Total operations
lu_count = count_lu + count_forward + count_backward

# Test code
print()
print("****** Operations Comparison Table ******")
print("-----------------------------------------")
print("Jacobi  | Gauss-Seidel | LU Factorization")
print("--------+--------------+-----------------")
print(f"{jacobi_count} | {gs_count}       | {lu_count}")
print("-----------------------------------------")
print()
```

**Implementation/Code:**
```python
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
```

# Leslie Matrices

## Leslie Matrices and the Power Method

**Routine Name:** leslie_power_method

**Description/Purpose:**
Finds the eigenvalue of a leslie matrix using a modified power method

**Input:** 
a leslie matrix, a vector in that can be multiplied by the leslie matrix, a tolerance for approximation, and a max number of iterations

**Output:**
the max eigenvalue of the leslie matrix

**Usage/Example:**
```python
# Example usage
leslie_matrix = [[0.1, 2.0, 1.5, 0.5], [0.5, 0, 0, 0], [0, 0.7, 0, 0], [0, 0, 0.9, 0]]
n = [1, 1, 1, 1]
tol = 1e-6
max_iter = 1000

max_value = leslie.leslie_power_method(leslie_matrix, n, tol, max_iter)
print("Largest Eigenvalue of Leslie:", max_value)
```
**Implementation/Code**
```python
def leslie_matrix_vector_multiply(leslie_matrix, vector):
    result = [0] * len(vector)
    # First row of Leslie matrix
    result[0] = sum(x * y for x, y in zip(leslie_matrix[0], vector))
    # Sub-diagonal elements
    for i in range(1, len(vector)):
        result[i] = leslie_matrix[i][i - 1] * vector[i - 1]
    return result

def leslie_power_method(leslie_matrix, n, tol, max_iter):
    b = normalize_vector.normalize_vector(n)
    for _ in range(max_iter):
        n = leslie_matrix_vector_multiply(leslie_matrix, b)
        b_next = normalize_vector.normalize_vector(n)
        if vector_norm.vector_norm(vector_subtract.vector_subtract(b_next, b)) < tol:
            break
        b = b_next
    max_value = sum(leslie_matrix_vector_multiply(leslie_matrix, b)[i] * b[i] for i in range(len(b))) / sum(b[i] * b[i] for i in range(len(b)))
    return max_value
```

## Leslie Matrices and Jacobian Iteration

**Routine Name:** jacobi_iteration (for leslie_jacobi.py)

**Description/Purpose:**
uses the jacobi method to provide new population vector with a multiplication modification. Note that otherwise this method could not work. Also there are other problems that arise with approximations and convergence not being guaranteed.

**Input:** 
a leslie matrix, an initial population vector, max iterations, and a tolerance for approximations

**Output:**
a new population vector

**Usage/Example:**
```python
leslie_matrix = np.array([[0, 2, 0.1],
                           [0.5, 0, 0],
                           [0, 0.4, 0]])

initial_population = np.array([100, 50, 25])

# Call the Jacobi iteration function to find the stable population distribution
stable_population = jacobi_iteration(leslie_matrix, initial_population)

# Print the result
print("Stable Population Distribution:")
print(stable_population)
```

**Implementation/Code**
```python
import numpy as np

def jacobi_iteration(leslie_matrix, initial_population, max_iterations=1000, tolerance=1e-6):
    # Initialize population vector
    population = initial_population.copy()
    
    for iteration in range(max_iterations):
        # Store the current population vector for comparison
        previous_population = population.copy()
        
        # Calculate the next generation's population
        population = leslie_matrix.dot(previous_population)
        
        # Check for convergence
        if np.linalg.norm(population - previous_population) < tolerance:
            break
    
    return population
```

# Matrix Generators

## Diagonally Dominant Matrix

**Routine Name:** create_diagonally_dominant_matrix

**Description/Purpose:** generates an NxN matrix that is diagonally dominant

**Input:** 
n - size
**Output:**
an NxN matrix
**Usage/Example:**
```python
# Generate a 100 by 100 diagonally dominant matrix and a random b vector
A = diagonal_matrix_generator.create_diagonally_dominant_matrix(n)
# generate n values for a b vector
b = [random.random() for _ in range(n)]
```

**Implementation/Code:**
```python
import random

def create_diagonally_dominant_matrix(n):
    """Create a diagonally dominant matrix of size n x n."""
    matrix = [[random.random() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        sum_of_row = sum(abs(matrix[i][j]) for j in range(n) if i != j)
        matrix[i][i] = sum_of_row + random.random() * 10
    return matrix
``` 
# openMP

## Jacobi Iteration Optimized
```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>  // Include OpenMP header

#define matrix_dimension 25

int n = matrix_dimension;
float sum;

int main()
{
    float A[n][n];
    float ones[n];
    float x0[n];
    float b[n];
    //
    // create a matrix
    //
    srand((unsigned int)(time(NULL)));
    float a = 5.0;
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            A[i][j] = ((float)rand()/(float)(RAND_MAX) * a);
        }
        x0[i] = ((float)rand()/(float)(RAND_MAX) * a);
    }
    //
    // modify the diagonal entries for diagonal dominance
    // --------------------------------------------------
    //
    for(int i=0; i<n; i++)
    {
        sum = 0.0;
        for(int j=0; j<n; j++)
        {
            sum = sum + fabs(A[i][j]);
        }
        A[i][i] = A[i][i] + sum;
    }
    //
    // generate a vector of ones
    // -------------------------
    //
    for(int j=0; j<n; j++)
    {
        ones[j] = 1.0;
    }
    //
    // use the vector of ones to generate a right hand side for the testing
    // operation in the code
    // ---------------------
    // 
    for(int i=0; i<n; i++)
    {
        sum = 0.0;
        for(int j=0; j<n; j++)
        {
            sum = sum + A[i][j];
        }
        b[i] = sum;
    }

    //
    // Jacobi iteration test
    // ---------------------
    //
    float tol = 0.0001;
    float error = 10.0 * tol;
    float x1[n];
    float res[n];
    int maxiter = 100;
    int iter = 0;

    // Parallelize this loop
    #pragma omp parallel for
    for(int i=0; i<n; i++)
    {
        sum = b[i];
        for(int j=0; j<n; j++)
        {
            sum = sum - A[i][j] * x0[i];
        }
        res[i] = sum;
    } 
    //
    // loop starts here for Jacobi
    // ---------------------------
    //
    while (error > tol && iter < maxiter) 
    {
        // Parallelize this loop
        #pragma omp parallel for
        for(int i=0; i<n; i++)
        {
            x1[i] = x0[i] + res[i] / A[i][i];
        }
        //
        // compute the error
        // -----------------
        //
        sum = 0.0;
        for(int i=0; i<n; i++)
        {
            float val = x1[i] - x0[i];
            sum = sum + val * val;
        }
        error = sqrt(sum);
        //
        // reset the input for the next loop through
        // -----------------------------------------
        //
        for(int i=0; i<n; i++)
        {
            x0[i] = x1[i];
        } 
        //
        // compute the next residual
        // -------------------------
        //
        // Parallelize this loop
        #pragma omp parallel for
        for(int i=0; i<n; i++)
        {
            sum = b[i];
            for(int j=0; j<n; j++)
            {
                sum = sum - A[i][j] * x0[j];
            }
            res[i] = sum;
        }
        //
        // update the iteration counter
        // ----------------------------
        //
        iter++;
        //
        // end of loop
        // -----------
    }

    for(int i=0; i<n; i++)
        printf("x[%d] = %6f \t res[%d] = %6f\n", i, x1[i], i, res[i]);

    return 0;
}

```

## The Power Method Optimized
```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 4  // Define the size of the matrix

void matrix_vector_multiply(double matrix[N][N], double vector[N], double result[N]) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        result[i] = 0.0;
        for (int j = 0; j < N; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

double dot_product(double vec1[N], double vec2[N]) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

double norm(double vector[N]) {
    return sqrt(dot_product(vector, vector));
}

void normalize(double vector[N]) {
    double vector_norm = norm(vector);
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        vector[i] /= vector_norm;
    }
}

int main() {
    double A[N][N] = {
        {4, 1, 2, 3},
        {0, 3, 4, 5},
        {0, 0, 1, 2},
        {0, 0, 0, 1}
    };
    double x[N] = {1, 1, 1, 1};  // Initial guess
    double y[N];
    double eigenvalue = 0.0;
    double tolerance = 1e-6;
    double error = 1.0;
    int max_iterations = 1000;
    int iteration = 0;

    while (error > tolerance && iteration < max_iterations) {
        matrix_vector_multiply(A, x, y);
        double new_eigenvalue = norm(y);
        error = fabs(new_eigenvalue - eigenvalue);
        eigenvalue = new_eigenvalue;
        normalize(y);
        for (int i = 0; i < N; i++) {
            x[i] = y[i];
        }
        iteration++;
    }

    printf("Dominant Eigenvalue: %lf\n", eigenvalue);
    return 0;
}
```





