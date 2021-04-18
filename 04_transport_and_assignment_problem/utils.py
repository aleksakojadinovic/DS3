import numpy as np

# TODO: Check if this can be refactored
def argmin_exclude(A, ex_rows, ex_columns):
    min_i = None
    min_j = None
    min_val = None

    for i, row in enumerate(A):
        for j, element in enumerate(row):
            if i in ex_rows or j in ex_columns:
                continue
            if min_val is None or element < min_val:
                min_i = i
                min_j = j
                min_val = element

    return min_i, min_j


def pack_indices(x, y, shape):
    return x*shape[1] + y

def unpack_index(x, shape):
    return np.unravel_index(x, shape)


def problem_matrix_to_cab(problem_matrix):
    problem_matrix = np.array(problem_matrix)
    C = problem_matrix[:-1, :-1]
    a = problem_matrix[:-1, -1]
    b = problem_matrix[-1, :-1]
    return C, a, b
