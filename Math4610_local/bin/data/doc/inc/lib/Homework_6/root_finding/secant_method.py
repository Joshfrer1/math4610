def f(x):
    return x**2 - 5*x + 6

def secant_method(x0, x1, tol):
    for _ in range(100):  # Assuming a max of 100 iterations
        if abs(f(x1) - f(x0)) < tol:
            raise ValueError('Division by zero in secant method')
        x_new = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x_new - x1) < tol:
            return x_new
        x0, x1 = x1, x_new
    return x1

# Example usage
tolerance = 1e-6
max_iterations = 100

# For the root near 2
root1 = secant_method(1.5, 2.5, tolerance, max_iterations)

# For the root near 3
root2 = secant_method(2.5, 3.5, tolerance, max_iterations)

print("Root near 2:", root1)
print("Root near 3:", root2)
