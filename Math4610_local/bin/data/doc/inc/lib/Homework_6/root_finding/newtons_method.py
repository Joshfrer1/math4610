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

# Example usage
tolerance = 1e-6
max_iterations = 100

# For the root near 2
root1 = newtons_method(2, tolerance, max_iterations)

# For the root near 3
root2 = newtons_method(3, tolerance, max_iterations)

print("Root near 2:", root1)
print("Root near 3:", root2)
