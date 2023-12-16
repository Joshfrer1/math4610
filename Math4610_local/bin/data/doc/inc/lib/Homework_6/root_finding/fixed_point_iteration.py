def f(x):
    return x**2 - 5*x + 6

def g1(x):
    return x + f(x)

def g2(x):
    return x - f(x)

def fixed_point_iteration(g, initial_guess, max_iter):
    x = initial_guess
    for i in range(max_iter):
        x_new = g(x)
        print(f"Iteration {i+1}: x = {x_new}")
        x = x_new
    return x

# Testing the iteration functions
initial_guess = 1.5
max_iterations = 10

print("Testing g1(x):")
fixed_point_iteration(g1, initial_guess, max_iterations)

print("\nTesting g2(x):")
fixed_point_iteration(g2, initial_guess, max_iterations)


def modified_g(x, epsilon):
    return x - epsilon * f(x)

def test_epsilon(epsilon, initial_guess, max_iter):
    print(f"Testing with epsilon = {epsilon}")
    fixed_point_iteration(lambda x: modified_g(x, epsilon), initial_guess, max_iter)

# Testing different values of epsilon
epsilons = [0.01, 0.1, 0.2, 0.5]
for epsilon in epsilons:
    test_epsilon(epsilon, initial_guess, max_iterations)
