import numpy as np

def leslie_jacobi_iteration(leslie_matrix, initial_population, max_iterations=1000, tolerance=1e-6):
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

