import numpy as np


def sign_zero(a):
    if a < 0:
        return -1
    else:
        return 1

def sign_eq(a, b):
    if np.isclose(a, 0.0) or np.isclose(b, 0.0):
        return True
    if a > 0 and b > 0:
        return True
    if a < 0 and b < 0:
        return True
    return False
    

def find_basic_sign_(matrix, sign):
    m, n = matrix.shape
    unit_column_indices     = []
    unit_column_row_indices = []
    for j in range(n):
        col = matrix[:, j]
        if len(col[np.isclose(col, 0)]) == m - 1:
            at_row = np.argwhere(np.invert(np.isclose(col, 0)))[0][0]
            if at_row in unit_column_row_indices:
                continue
            if not sign_eq(col[at_row], sign):
                continue
            unit_column_indices.append(j)
            unit_column_row_indices.append(at_row)

    return unit_column_indices, unit_column_row_indices


def check_for_duality(simplex_matrix):
    A, b, c, _ = unpack_matrix(simplex_matrix)
    return check_for_duality_abc(c, A, b)

def check_for_duality_abc(c, eqA, eqb):
    if not (eqb <= 0).any():
        return False, [], []

    m, n = eqA.shape
    basic_col_indices, basic_row_indices = find_basic_sign_(eqA, -1)
    if len(basic_col_indices) == m and np.isclose(c[basic_col_indices], 0.0).all():
        return True, basic_col_indices, basic_row_indices

    basic_col_indices, basic_row_indices = find_basic_sign_(eqA, +1)
    if len(basic_col_indices) == m and np.isclose(c[basic_col_indices], 0.0).all():
        return True, basic_col_indices, basic_row_indices

    return False, [], []

def pack_to_matrix(A, b, c):
    m, n = A.shape
    matrix = np.zeros((m+1, n+1))
    matrix[:m, :n] = A
    matrix[-1, :-1] = c
    matrix[:-1, -1] = b
    return matrix

def unpack_matrix(matrix):
    A = matrix[:-1, :-1]
    b = matrix[:-1, -1]
    c = matrix[-1, :-1]
    F0 = matrix[-1, -1]
    return A, b, c, F0

def find_basic_columns(eqA, eqb):
    m, n = eqA.shape
    unit_column_indices     = []
    unit_column_row_indices = []
    for j in range(n):
        col = eqA[:, j]
        if len(col[np.isclose(col, 0)]) == m - 1:
            at_row = np.argwhere(np.invert(np.isclose(col, 0)))[0][0]
            if at_row in unit_column_row_indices:
                continue
            # if sign_zero(col[at_row]) != sign_zero(eqb[at_row]):
            #     continue
            if not sign_eq(col[at_row], eqb[at_row]):
                continue
            unit_column_indices.append(j)
            unit_column_row_indices.append(at_row)

    return unit_column_indices, unit_column_row_indices


def swap_basis(simplex_matrix, basic_indices, i0, j0):
    basic_indices = np.array(basic_indices)

    found = False    

    for ii, basic_index in enumerate(basic_indices):
        
        curr_basic_column = simplex_matrix[:-1, basic_index].copy()

        curr_basic_non_zero = np.invert(np.isclose(curr_basic_column, 0))
        curr_non_zero_idx_idces = np.argwhere(curr_basic_non_zero)[0]
        curr_non_zero_idx = curr_non_zero_idx_idces[0]
        
        if curr_non_zero_idx == i0:
            found = True
            basic_indices[ii] = j0
            break

    return list(basic_indices)

def fetch_sol_from_simplex_matrix(simplex_matrix, basic_indices):   
    m, n = simplex_matrix.shape

    solution = np.zeros(n-1)
    for j in basic_indices:
        row_idx = np.argwhere(np.invert(np.isclose(simplex_matrix[:-1, j], 0)))[0][0]
        coeff = simplex_matrix[row_idx, j]
        solution[j] = simplex_matrix[row_idx, -1] / coeff

    return solution

