from linear_systems import back_substitution, forward_elimination

def gaussian_elimination(A, b):
    A, b = forward_elimination.forward_elimination(A, b)
    return back_substitution.back_substitution(A, b)