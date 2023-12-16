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

    