import numpy as np


def sign_zero(a):
    if a < 0:
        return -1
    else:
        return 1

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
            if sign_zero(col[at_row]) != sign_zero(eqb[at_row]):
                continue
            unit_column_indices.append(j)
            unit_column_row_indices.append(at_row)

    return unit_column_indices, unit_column_row_indices