import secant_method

def f(x):
    return x**2 - 5*x + 6

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

# Example usage
intervals = [(1, 2.5), (2.5, 4)]
tolerance = 0.0001

roots = []
for interval in intervals:
    root = hybrid_method(interval, tolerance)
    if root is not None:
        roots.append(root)

print("Roots:", roots)
