def f(x):
    return x**2 - 5*x + 6

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

# Example usage
intervals = [(1, 2.5), (2.5, 4)]  # Adjusted intervals
tolerance = 0.001

roots = []
for interval in intervals:
    root = bisection_method(interval[0], interval[1], tolerance)
    if root is not None:
        roots.append(root)

print("Roots:", roots)

