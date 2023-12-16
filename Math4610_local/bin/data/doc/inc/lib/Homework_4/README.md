# Math 4610 Fundamentals of Computational Mathematics Software Manual 

**Author:** Joshua Frerichs

**Language:** C, using gcc compiler

# Table of Contents
1. [Machine Epsilon of a Float](#machine-epsilon-of-a-float)
2. [Machine Epsilon of a Double](#machine-epsilon-of-a-double)
3. [L-2 Normal Vector](#l-2-normal-vector)
4. [L-1 Normal Vector](#l-1-normal-vector)
5. [Infinity Normal Vector](#infinity-normal-vector)
6. [L-2 Distance](#l-2-distance)
7. [L-1 Distance](#l-1-distance)
8. [Infinity Distance](#infinity-distance)
9. [Linear Regression using Least Squares Method](#linear-regression-using-least-squares-method)
10. [The Forward Difference Quotient](#the-forward-difference-quotient)
11. [The Backward Difference Quotient](#the-backward-difference-quotient)
12. [Upper Triangular Matrix Reducer](#upper-triangular-matrix-reducer)
13. [Systems of Equations Solver](#systems-of-equations-solver)


## Machine Epsilon of a Float
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

## Machine Epsilon of a Double
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

## L-2 Normal Vector
      
**Function Name:**          l_2_norm

For example,

    gcc l_2_norm.c -o l_2_norm

will produce a .o file such that you can run:

    ./l_2_norm

**Description/Purpose:**

This function takes each element of a vector, squares it, sums those elements, and returns the square root of that sum.

**Input:** The funtion takes a vector and a size as parameters. The size is required in order to iterate through the vector. The vector is also a pointer to a predefined vector such that I can pass in a vector of any size I want.

**Output:** Returns the L2 normal vector as a result.

**Usage/Example:**

The function takes a pointer to vector and a size to be able to iterate through the vector. You can call this function in a main function through the shared libary. You must predefine an array to act as your vector. The size can then be set as

    sizeof(vector)/sizeof(double)

This is done to help the size become dynamic in nature. In C you have to define your elements regardless of whether its a pointer or a dynamic array. I chose to refrain from managing memory and used a static array with a dynamic size for ease of use.

    double l_2_norm(double *vector, int size)

This is designed such that the values inputted can be dynamic. 

**Implementation/Code:**
We must use the math libary to have access to the pow() and sqrt() functions.

    #include <math.h>

Again, we take a pointer to an array and a size

    double l_2_norm(double *vector, int size) {

The we initialize a length

        double length = 0;

The for loop is straigtforward, from 0 to the size of the array

        for(int i = 0; i < size; i++) {
            length += pow(vector[i], 2);
        }

We then summate the values of i sqaured from the vector. Then we print out the values using our test arrays and return the square root of that length.

        printf("%lf\n", sqrt(length));
        return sqrt(length); 
    }

## L-1 Normal Vector
**Function Name:**           l_1_norm

For example,

    gcc l_1_norm.c -o l_1_norm

will produce a .o file such that you can run:

    ./l_1_norm

**Description/Purpose:** The function takes the absolute value of all the elements in a vector and sums them.

**Input:** The funtion takes a vector and a size as parameters. The size is required in order to iterate through the vector. The vector is also a pointer to a predefined vector such that I can pass in a vector of any size I want.

**Output:** Returns the L1 normal vector as a result.

**Usage/Example:**
The function takes a pointer to vector and a size to be able to iterate through the vector. You can call this function in a main function through the shared libary. You must predefine an array to act as your vector. The size can then be set as

    sizeof(vector)/sizeof(double)

This is done to help the size become dynamic in nature. In C you have to define your elements regardless of whether its a pointer or a dynamic array. I chose to refrain from managing memory and used a static array with a dynamic size for ease of use.

    double l_1_norm(double *vector, int size)

This is designed such that the values inputted can be dynamic. 

**Implementation/Code:**

We need the math library to use fabs() which acts the as the abs() function for floats and doubles.

    #include <math.h>

Define your function and pass in a pointer to a vector and its size dynamically

    double l_1_norm(double *vector, int size) {

Initialize a length

        double length = 0;

A simple for loop that goes from 0 to the size of the vector

        for(int i = 0; i < size; i++) {

Summate the absolute value of each element in the vector

            length += fabs(vector[i]);
        }

Print out our length and return it

        printf("%lf\n", length);
        return length;
    }

## Infinity Normal Vector
**Function Name:**           infinity_norm

For example,

    gcc linf_norm.c -o linf_norm

will produce a .o file such that you can run:

    ./linf_norm

**Description/Purpose:** This function sets each element in the vector to be an absolute value, and then uses a maximum sort to find the max value in the vector.

**Input:** The funtion takes a vector and a size as parameters. The size is required in order to iterate through the vector. The vector is also a pointer to a predefined vector such that I can pass in a vector of any size I want.

**Output:** Returns the infinity normal vector as a result.

**Usage/Example:** The function can be called in a main function from the shared library:

    double infinity_norm(double *vector, int size)

Again, a pointer to a vector and a size as to help iterate through the vector

**Implementation/Code:**

We need the math library again for fabs()

    #include <math.h>

A simple definition that takes a pointer to a vector and a size

    double infinity_norm(double *vector, int size) {

Here we have to do some comparisons to find a max. To start, lets initialize a max variable set to the first element in the vector.
        
        double max = vector[0];

Then we can iterate through the arrays size

        for(int i = 0; i < size; i++) {

We then make each element in the vector absolute

            double abs_i = fabs(vector[i]);

Compare the intial value to the elements in the array. If the max is smaller than the value its being compared to, the larger value becomes the new max until we find the largest element.

            if (max < abs_i) {
                max = abs_i;
            }
        }

Return the new max.

        return max;
    }

## L-2 Distance
**Function Name:**           l_2_distance

For example,

    gcc l2_distance.c -o l2_distance

will produce a .o file such that you can run:

    ./l2_distance

**Description/Purpose:** The function iterates through two vectors and summates the squared difference of u[i] and v [i] and then returns the square root of that distance.

**Input:** The functions takes two vectors, u and v, and a size. The size is required in order to iterate through the vectors. The vectors are pointers to predefined vectors such that I can pass in a vector of any size I want.

**Output:** Returns the L2 Distance as a result.

**Usage/Example:**

You can simply call it in a main function with the shared library

    double l_2_distance(double *u, double *v, int size)

The output will depend on the array the user defines. Again, designed to be dynamic in size such that any two vectors of the same length will work

    double u[] = {1, 2, 3};
    double v[] = {4, 5, 6};

**Implementation/Code:**

We will need the math library for pow() and sqrt()

    #include <math.h>

Simple definition that takes two pointer to vectors and a size

    double l_2_distance(double *u, double *v, int size) {

Initialize a distance

        double distance = 0;

We only need one loop since the vectors must be the same size

        for(int i = 0; i < size; i++) {

Summate the sqaured difference of u[i] and v[i] where the i's are designed, mapped elements.

            distance += pow((u[i] - v[i]), 2);
        }

Print out the result from out test and return the square root of the distance

        printf("%lf\n", sqrt(distance));
        return sqrt(distance);
    }

## L-1 Distance
**Function Name:**           l_1_distance

For example,

    gcc l1_distance.c -o l1_distance

will produce a .o file such that you can run:

    ./l1_distance

**Description/Purpose:** Takes two vectors and iterates through them and summates the absolute difference of the values of u[i] and v[i]. The value is returned as a distance.

**Input:** The functions takes two vectors, u and v, and a size. The size is required in order to iterate through the vectors. The vectors are pointers to predefined vectors such that I can pass in a vector of any size I want.

**Output:** Returns the L1 distance as a result.

**Usage/Example:**

Super simple, just call the function in a main function

    double l_1_distance(double *u, double *v, int size)

The output will depend on the array the user defines. Again, designed to be dynamic in size such that any two vectors of the same length will work

    double u[] = {1, 2, 3};
    double v[] = {4, 5, 6};

**Implementation/Code:**

We include the math library to use fabs()

    #include <math.h>

    double l_1_distance(double *u, double *v, int size) {
        double distance = 0;
        for(int i = 0; i < size; i++) {
            distance += fabs(u[i] - v[i]);
        }
        printf("%lf", distance);
        return distance;
    }

## Infinity Distance
**Function Name:**           infinity_norm_distance

For example,

    gcc linf_distance.c -o linf_distance

will produce a .o file such that you can run:

    ./linf_distance

**Description/Purpose:**

**Input:** Takes two vectors, u and v, and a size. The size is required in order to iterate through the vectors. The vectors are pointers to predefined vectors such that I can pass in a vector of any size I want.

**Output:** Returns the infinity distance as a result.

**Usage/Example:** Uses the [infinity norm](#infinity-normal-vector) as a helper function such that the distance can be computed by find the infinity norms of two vectors and finding their absolute difference.

You can simply call the function in main

double infinity_norm_distance(double u[], double v[], int size)

So a seemingly strange thing happens with this function call because we cannot pass in two arrays by pointers. Instead we can define two double typed arrays as normal, they get passed in to be used as pointers for our calls to infinity norm. Basically we point to them through our calls to infinity_norm() to then compute the difference between the two maxes.

The output again is dynamic because we like to be able to take in any vectors we want.

**Implementation/Code:**

The function call takes in two double typed arrays which act as an easy way to pass in functions that return doubles. We need a size so the infinity_norm functions can iterate through to find the two maxes to compare.

    double infinity_norm_distance(double u[], double v[], int size) {

set a compute variable to take the difference of two maxes

    double compute = infinity_norm(u, size) - infinity_norm(v, size);

return the absolute value of the difference

    return fabs(compute);
}
## Linear Regression using Least Squares Method
**Function Name:**           least_squares

For example,

    gcc linreg.c -o linreg

will produce a .o file such that you can run:

    ./linreg

**Description/Purpose:** Returns the function for the best fit line

**Input:** Takes a size (of our data points), an x array, a y array, a pointer to an m, and a pointer to a b. (This is useful for testing)

**Output:** 

We can display the output in main. Again using pointers so we can keep track of the values of m and b

    double m, b;
    least_sqaures(sizeof(u)/sizeof(double), u, v, &m, &b);
    printf("Best fit line: y = %lf x + %lf\n", m, b);

**Usage/Example:**

Again, designed to be dynamic so the usage can be whatever example we want. Refer to output to see how we use it.

**Implementation/Code:**

The function takes n data points, all the x values, all the y values, and our slope m, and intercept b

    void least_sqaures(int n, double x[], double y[], double* m, double* b) 
    
Initialize the sum of x, sum of y, some of the two multiplied, and sum of x squared per the definition of linear regression using least squares    
    
    {
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
        
Iterate through out points to get the sums

        for (int i = 0; i < n; i++) {
            sum_x += x[i];
            sum_y += y[i];
            sum_xy += x[i] * y[i];
            sum_x2 += x[i] * x[i];
        }
        
Find our slope

        *m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

Find our intercept

        *b = (sum_y - *m * sum_x) / n;
    }

## The Forward Difference Quotient
**Function Name:**           forward_difference

For example,

    gcc forward_difference.c -o forward_difference

will produce a .o file such that you can run:

    ./forward_difference

**Description/Purpose:** Use the definition of the forward difference quotient to approximate the derivative of a function

**Input:** Takes a double x which is the return value of a predefined function x^3. Also takes a double h which is our increment. We are using .000001 for our test. We will set x = 3

**Output:** 

We simply print out the return value in main

    printf("%f\n", backward_difference(3, .000001));

**Usage/Example:**

x^3 = 27. 3x^2 = 27. My output is:

    26.999991

**Implementation/Code:**

Simply pass in a value and return that value cubed

    static double fx_x_cubed(double x) {
        return x * x * x;
    }

pass in any x value and pass in an h close to zero. Return the definition of the forward difference quotient.

    double forward_difference(double x, double h) {
        return (fx_x_cubed(x + h) - fx_x_cubed(x))/h;
    }

## The Backward Difference Quotient
**Function Name:**           backward_difference

For example,

    gcc backward_difference.c -o backward_difference

will produce a .o file such that you can run:

    ./backward_difference

**Description/Purpose:** Use the definition of the forward difference quotient to approximate the derivative of a function

**Input:**
Takes a double x which is the return value of a predefined function x^3. Also takes a double h which is our increment. We are using .000001 for our test. We will set x = 3

**Output:** 

simple print statement in main

    printf("%f\n", forward_difference(3, .000001));

**Usage/Example:**

x^3 = 27. 3x^2 = 27. My output is:

    27.000009

**Implementation/Code:**

A simple helper function to test with

    static double fx_x_cubed(double x) {
        return x * x * x;
    }

Return the definition of the backward difference quotient.

    double backward_difference(double x, double h) {
        return (fx_x_cubed(x) - fx_x_cubed(x - h))/h;
    }

## Upper Triangular Matrix Reducer and Systems Solver
**Function Name:**           upper_triangular

For example,

    gcc alu.c -o alu

will produce a .o file such that you can run:

    ./alu

**Description/Purpose:** Takes a non zero matrix and reduces it to upper triangular form using Gaussian elimination

**Input:** 

Takes a multidimensional array with an augmented matrix

**Output:** 

Does not output the triangle since it was not a requirement. What it allows us to do is pass the matrix to a back_substitution function to solve the matrix.

**Usage/Example:**

Here is a test matrix that uses diagonal dominance for ease of use.

    double mat[N][N + 1] = {
        {3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 2},
        {0, 1, 5, 1, 0, 0, 0, 0, 0, 0, 3},
        {0, 0, 1, 6, 1, 0, 0, 0, 0, 0, 4},
        {0, 0, 0, 1, 7, 1, 0, 0, 0, 0, 5},
        {0, 0, 0, 0, 1, 8, 1, 0, 0, 0, 6},
        {0, 0, 0, 0, 0, 1, 9, 1, 0, 0, 7},
        {0, 0, 0, 0, 0, 0, 1, 10, 1, 0, 8},
        {0, 0, 0, 0, 0, 0, 0, 1, 11, 1, 9},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 12, 10}
    };
    double x[N];

    upper_triangular(mat, sizeof(mat)/sizeof(double));

**Implementation/Code:**

Define a constant 10 for out test case

    #define N 10

Simple function declaration. Since you cannot pass multidimensional arrays by pointers, we have to define the size of our array and pass by reference.

    void upper_triangular(double a[N][N + 1]) {

the outer k loop represents goes represents our current pivot row

        for (int k = 0; k < N - 1; k++) {

middle i loop represents rows below pivot rows

            for (int i = k + 1; i < N; i++) {

Inside the middle loop, the factor for each row i is calculated as the ratio of the i-th row's k-th column element to the pivot element a[k][k].

                double factor = a[i][k] / a[k][k];

The innermost loop with index j goes from k to N and updates each element in row i by subtracting the scaled version of the pivot row. This effectively aims to make all values below the pivot in column k become zero.

                for (int j = k; j < N + 1; j++) {
                    a[i][j] -= factor * a[k][j];
                }
            }
        }
    }


## Systems of Equations Solver

**Function Name:**           upper_triangular

For example,

    gcc alu.c -o alu

will produce a .o file such that you can run:

    ./alu

**Description/Purpose:** Uses back subsitution to solve a linear system equations derived from an upper triangular matrix

**Input:** Takes a multidimensional array of size N and N + 1 and array x to store solutions

**Output:** Outputs solutions to the matrix passed in to it

**Usage/Example:**

    upper_triangular(mat, sizeof(mat)/sizeof(double));
        back_substitution(mat, x, sizeof(mat)/sizeof(double));
        printf("Solution:\n");
        for (int i = 0; i < N; i++)
            printf("x%d = %f\n", i + 1, x[i]);

Solutions using 10x10 matrix above:

    x1 = 0.221119
    x2 = 0.336644
    x3 = 0.432307
    x4 = 0.501821
    x5 = 0.556769
    x6 = 0.600799
    x7 = 0.636842
    x8 = 0.667622
    x9 = 0.686935
    x10 = 0.776089

**Implementation/Code:**

Take a N by N + 1 array (for the augmented matrix)

    void back_substitution(double a[N][N+1], double x[N]) {

outer loop starts from the bottom and work our way up. x[i] is set to our most right hand element

        for (int i = N - 1; i >= 0; i--) {
            x[i] = a[i][N];

inner loop with index j iterates over the columns to the right of the diagonal for the current row i. For each of these columns, the code subtracts the product of the i-th row's coefficient and the already computed solution component from x[i]. The purpose of this subtraction is to account for the contributions of known variables to the right-hand side of the equation.

            for (int j = i + 1; j < N; j++) {
                x[i] -= a[i][j] * x[j];
            }

After inner loop completes for a row i, x[i] contains the sum of terms for the variables to the right of the current variable. To isolate it, x[i] is divied by the diagonal element of the i-th row

            x[i] /= a[i][i];
        }
    }

x[] will contain the solution to the system of equations.
