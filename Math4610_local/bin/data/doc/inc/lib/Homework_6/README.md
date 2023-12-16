# Math 4610 Fundamentals of Computational Mathematics Software Manual Template File
This is a template file for building an entry in the student software manual project. You should use the formatting below to
define an entry in your software manual.

<hr>

**Author:** Joshua Frerichs

**Language:** Python 3, using python3 interpreter

For example,

    python3 <file_name.py>

**Routine Name:**           bisection_method

**Description/Purpose:** 
Uses the bisection method to find roots

**Input:** 
takes in a and b which represent an interval of where we want to test. Also a tolerance for how close we want to be with our roots

**Output:** 
outputs a root. to find multiple roots you would test on a variety of intervals represented as tuples in the test code.

**Usage/Example:**

intervals = [(1, 2.5), (2.5, 4)]  # Adjusted intervals
tolerance = 0.001

roots = []
for interval in intervals:
    root = bisection_method(interval[0], interval[1], tolerance)
    if root is not None:
        roots.append(root)

print("Roots:", roots)

**Implementation/Code:** 
def bisection_method(a, b, tol):
    # Check if the function f changes sign over the interval [a, b].
    # This is a necessary condition for the bisection method to work,
    # as it relies on the Intermediate Value Theorem.
    if f(a) * f(b) >= 0:
        print("No root found in the interval [", a, ",", b, "]")
        return None

    # Repeat the following steps until the size of the interval [a, b]
    # is reduced to the specified tolerance 'tol'.
    while (b - a) / 2.0 > tol:
        # Compute the midpoint of the current interval.
        midpoint = (a + b) / 2.0

        # Check if the midpoint is a root of the function f.
        if f(midpoint) == 0:
            return midpoint  # If yes, return the midpoint as the root.

        # If f(midpoint) is not zero, determine which subinterval to choose
        # for the next step. This is done based on where the sign change occurs.

        # If f(a) and f(midpoint) have different signs, the root must be in [a, midpoint].
        elif f(a) * f(midpoint) < 0:
            b = midpoint  # Update b to be the midpoint.
        
        # If f(a) and f(midpoint) have the same sign, the root must be in [midpoint, b].
        else:
            a = midpoint  # Update a to be the midpoint.

    # Once the interval size is smaller than the tolerance, return the average
    # of a and b as the estimated root.
    return (a + b) / 2.0


**Routine Name:**           newtons_method

**Description/Purpose:** 
uses newton's method to find roots for a funtion

**Input:** 
the inputs are an initial guess, tolerance, and max iterations

**Output:** 
A single root

**Usage/Example:**
max_iterations = 100

For the root near 2
root1 = newtons_method(2, tolerance, max_iterations)

For the root near 3
root2 = newtons_method(3, tolerance, max_iterations)

print("Root near 2:", root1)
print("Root near 3:", root2)

**Implementation/Code:**
def newtons_method(initial_guess, tol, max_iter):
    # Initialize the current approximation with the initial guess.
    x = initial_guess

    # Repeat the Newton-Raphson iteration for a maximum of 'max_iter' times.
    for _ in range(max_iter):
        # Update the approximation using the Newton-Raphson formula:
        # x_new = x - f(x) / df(x)
        # Here, f(x) is the function whose root we are trying to find,
        # and df(x) is the derivative of f(x).
        x_new = x - f(x) / df(x)

        # Check if the difference between the new and old approximations
        # is less than the specified tolerance 'tol'. If it is, the method
        # has converged to a root, and we return the current approximation.
        if abs(x_new - x) < tol:
            return x_new

        # Update the current approximation with the new approximation.
        x = x_new

    # If the method did not converge within 'max_iter' iterations,
    # return the last approximation.
    return x



**Routine Name:**           secant_method

**Description/Purpose:** 
Uses the secant method to find roots of a function

**Input:** 
x0, x1 as an interval, and a tolerance

**Output:** 
a root found by the secant method

**Usage/Example:**
tolerance = 1e-6
max_iterations = 100

For the root near 2
root1 = secant_method(1.5, 2.5, tolerance, max_iterations)

For the root near 3
root2 = secant_method(2.5, 3.5, tolerance, max_iterations)

print("Root near 2:", root1)
print("Root near 3:", root2)

**Implementation/Code:**
def secant_method(x0, x1, tol):
    # Perform the secant method iterations, assuming a maximum of 100 iterations.
    for _ in range(100):
        # Check for a potential division by zero error. This occurs when the function
        # values at x0 and x1 are too close, which can happen if x0 and x1 are very close
        # to each other or if they are both close to a root. 
        if abs(f(x1) - f(x0)) < tol:
            raise ValueError('Division by zero in secant method')

        # Calculate the next approximation, x_new, using the secant method formula:
        # x_new = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        # This formula uses the values of the function at x0 and x1 to approximate
        # the slope of the secant line and find the x-intercept of this line,
        # which becomes the next approximation of the root.
        x_new = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))

        # Check if the difference between the new approximation and the last
        # approximation is less than the specified tolerance. If it is, the method
        # has converged to a root, and we return the current approximation.
        if abs(x_new - x1) < tol:
            return x_new

        # Update the previous two approximations for the next iteration.
        x0, x1 = x1, x_new

    # If the method did not converge within 100 iterations, return the last approximation.
    return x1


**Routine Name:**           hybrid_method

**Description/Purpose:** 
uses the bisection method to find intervals of where roots could be and the secant method to narrow down faster finding the roots location

**Input:** 
An interval and a tolerance. Note that the bisection method was changed to return a location instead of a root. Still using the same secant method from before.

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

**Output:** 
the roots being looked for

**Usage/Example:**
intervals = [(1, 2.5), (2.5, 4)]
tolerance = 0.0001

roots = []
for interval in intervals:
    root = hybrid_method(interval, tolerance)
    if root is not None:
        roots.append(root)

print("Roots:", roots)

**Implementation/Code:** 
def hybrid_method(interval, tol):
    # First, use the bisection method to narrow down the interval where the root is likely to be.
    # The function 'bisection_for_interval' is called with the provided interval and tolerance.
    # It returns a narrowed interval [a, b] where a root is suspected to be present.
    a, b = bisection_for_interval(interval[0], interval[1], tol)

    # If 'bisection_for_interval' returns None, it means that it couldn't find a suitable interval
    # where a root might be located (likely because there was no sign change in f(x) over the
    # original interval). In this case, return None to indicate that no root was found.
    if a is None:
        return None

    # Check if the bisection method has already found the root. This can happen if the
    # size of the interval [a, b] becomes smaller than the tolerance, suggesting that
    # both a and b are very close to the root. If so, return this value as the root.
    if a == b:
        return a

    # If the root was not precisely determined by the bisection method, use the secant method
    # for faster convergence. The secant method is called with the narrowed interval [a, b]
    # and the given tolerance. It's expected that the secant method will now converge quickly
    # to the root within this narrowed interval.
    return secant_method(a, b, tol)

