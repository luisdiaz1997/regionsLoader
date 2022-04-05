import numpy as np

def from_upper_triu(vector_repr, matrix_len): #From basenji
    z = np.zeros((matrix_len,matrix_len))
    triu_tup = np.triu_indices(matrix_len, 0)
    diag_tup = np.diag_indices(matrix_len)
    z[triu_tup] = vector_repr
    z = z + z.T
    z[diag_tup] = z[diag_tup]/2
    return z