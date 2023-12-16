from vector_functions import vector_norm

def normalize_vector(vector):
    norm = vector_norm.vector_norm(vector)
    return [x / norm for x in vector]