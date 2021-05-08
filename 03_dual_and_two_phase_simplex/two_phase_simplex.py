import numpy as np
from regular_simplex import reg_simplex as r_simplex
import lp_input_parser as lparse
import sys
from scipy.optimize import linprog

FLOAT_T = 'float32'

def print_constraints(A, b, signs, index_from_zero=True):
    var_range = range(len(A[0])) if index_from_zero else range(1, len(A[0]) + 1)
    var_names = [f'x_{i}' for i in var_range]
    for row, b_val, sign in zip(A, b, signs):
        lhs = ' + '.join(f'{r}*{varname}' for r, varname in zip(row, var_names))
        rhs = str(b_val)
        print('\t' + lhs + ' ' + sign + ' ' + rhs)


# Assumes all dimensions are asserted
# Checks which basic columns are missing
# modifies matrix
def prep_for_phase_one(eqA):
    m, n = eqA.shape

    basic_column_indices = []
    basic_missing = [i for i in range(m)]
    for j in range(n):
        column = eqA[:, j]
        is_basic = len(column[column == 1]) == 1 and len(column[column == 0]) == len(column) - 1
        if is_basic:
            basic_column_indices.append(j)
            basic_missing.remove(np.nonzero(column)[0])



    
    new_basic_columns = np.array([[1.0 if k == missing_index else 0.0 for k in range(m)] for missing_index in basic_missing])
    for column in new_basic_columns:
        eqA = np.hstack((eqA, column.reshape(-1, 1)))

    artificial_indices = [i for i in range(n, eqA.shape[1])]
    all_basis_indices = basic_column_indices + artificial_indices

    return eqA, artificial_indices, all_basis_indices
    


# We add m artifical variables and we do not care if some other were basic
# I have no idea whether this will work
def simple_prep(eqA):
    m, n = eqA.shape
    subA = np.zeros((m, n + m))

    subA[:m, :n] = eqA
    subA[:, n:] = np.eye(m)

    return subA, list(range(n, n + m))

def adv_prep(eqA, eqb):
    eqA = np.array(eqA, dtype=FLOAT_T)
    eqb = np.array(eqb, dtype=FLOAT_T)
    
    m, n = eqA.shape
    unit_column_indices     = []
    unit_column_row_indices = []
    for j in range(n):
        col = eqA[:, j]
        if len(col[col == 0]) == m - 1:
            at_row = np.argwhere(col != 0)[0][0]
            if at_row in unit_column_row_indices:
                continue
            unit_column_indices.append(j)
            unit_column_row_indices.append(at_row)

    if len(unit_column_indices) == m:
        return eqA, eqb, unit_column_indices, [], []

    
    # Divide them all by their coefficients in order for them to be truly unit
    for row_idx, col_idx in zip(unit_column_row_indices, unit_column_indices):
        coeff = eqA[row_idx, col_idx]
        eqA[row_idx] /= coeff
        eqb[row_idx] /= coeff


    artificial_unit_row_indices        = [i for i in range(m) if i not in unit_column_row_indices]
    artificial_unit_column_indices  = []
    for missing_row_index in artificial_unit_row_indices:
        artificial_column = np.zeros(m)
        artificial_column[missing_row_index] = 1.0
        eqA = np.append(eqA, artificial_column.reshape(-1, 1), axis=1)
        artificial_unit_column_indices.append(eqA.shape[1] - 1)

    num_artif = len(artificial_unit_row_indices)

    return eqA, eqb, unit_column_indices + artificial_unit_column_indices, artificial_unit_column_indices, artificial_unit_row_indices

    
    
        




    


# Construct sub-problem matrix by adding artificial
# objective funcion and do the pivoting thingy
def get_sub_simplex_matrix(subA, eqb, artificial_indices, artificial_row_indices):

    m, n = subA.shape
    num_artif = len(artificial_indices)
    # m + 1 for the new objective function
    # n + 1 for the b
    sub_simplex_matrix = np.zeros((m + 1, n + 1))
    sub_simplex_matrix[:m, :n] = subA
    sub_simplex_matrix[:-1, -1] = eqb
    sub_simplex_matrix[-1, n - num_artif:-1] = np.ones(num_artif)
    


    for ar in artificial_row_indices:
        sub_simplex_matrix[-1, :] -= sub_simplex_matrix[ar, :]


    return sub_simplex_matrix

def piv_artif(sub_simplex_matrix, artificial_indices):
    for i, artifical_idx in enumerate(artificial_indices):
        sub_simplex_matrix[-1, :] -= sub_simplex_matrix[i, :]

    return sub_simplex_matrix

def fix_indices(index_collection, removed_index):
    for i in range(len(index_collection)):
        if index_collection[i] >= removed_index:
            index_collection[i] -= 1

    return index_collection

def find_basic_columns(mat):
    bs = []
    for i in range(mat.shape[1]):
        col = mat[:, i]
        if len(col[col == 1]) == 1 and len(col[col == 0]) == len(col) - 1:
            bs.append(i)

    return bs

def remove_columns_and_fix_index_lists(matrix, cols_to_delete, index_lists):
    for j, col_to_delete in enumerate(cols_to_delete):
        for index_list in index_lists:
            for i in range(len(index_list)):
                if index_list[i] >= col_to_delete:
                    index_list[i] -= 1

        for k in range(j+1, len(cols_to_delete)):
            if cols_to_delete[k] >= col_to_delete:
                cols_to_delete[k] -= 1

        matrix = np.delete(matrix, [col_to_delete], axis=1)

    return matrix, index_lists

# Assumes all constrains have been converted to equalities
def two_phase_simplex_solver(c, eqA, eqb):
    c   = np.array(c, dtype=FLOAT_T)
    eqA = np.array(eqA, dtype=FLOAT_T)
    eqb = np.array(eqb, dtype=FLOAT_T)

    n = len(c)
    if n != eqA.shape[1]:
        raise ValueError(f'Length of objective function is {n} but width of eqA is {eqA.shape[1]}')

    m = eqA.shape[0]
    if m != len(eqb):
        raise ValueError(f'eqA has {m} constraints but b-vector has {len(eqb)} values')


    eqA, eqb, basic_indices, artificial_indices, artificial_row_indices = adv_prep(eqA, eqb)

    # TODO: If no artificial just use run simplex!

    subA = get_sub_simplex_matrix(eqA, eqb, artificial_indices, artificial_row_indices)

    sub_m, sub_n = subA.shape
    sub_b_vector = subA[:-1, -1]
    x0length = sub_n - 1
    x0 = np.append(np.zeros(x0length - len(sub_b_vector)), sub_b_vector)

    indicator, last_matrix, last_basic_indices = r_simplex(subA, basic_indices)

    if not indicator:
        print(f'Phase one no solution.')
        return None

    phase_one_opt = last_matrix[-1, -1]

    if not np.isclose(phase_one_opt, 0.0):
        print(f'Phase one opt is nonzero, no solution then.')
        print(f'it is: {phase_one_opt}')
        return None



    # first we remove all artificial variables that are not basic
    cols_to_delete = []
    for artif_index in artificial_indices:
        if artif_index in last_basic_indices:
            continue
        cols_to_delete.append(artif_index)

    artificial_indices = [a for a in artificial_indices if a not in cols_to_delete]
    last_basic_indices = [a for a in last_basic_indices if a not in cols_to_delete]

    new_matrix, [artificial_indices, last_basic_indices] = remove_columns_and_fix_index_lists(last_matrix, cols_to_delete, [artificial_indices, last_basic_indices])

    cols_to_delete = []
    for art_and_basic in artificial_indices:
        column = new_matrix[:, art_and_basic]
        row_idx = np.argwhere(new_matrix[:, art_and_basic] == 1)[0][0]
        row = new_matrix[row_idx]
        
        if (len(row[row == 0]) == len(row) - 1):

            # TODO: remove this row and this column
            artificial_indices.remove(art_and_basic)
            last_basic_indices.remove(art_and_basic)
            cols_to_delete.append(art_and_basic)
            new_matrix = np.delete(new_matrix, [row_idx], axis=0)

 
    new_matrix, [artificial_indices, last_basic_indices] = remove_columns_and_fix_index_lists(new_matrix, cols_to_delete, [artificial_indices, last_basic_indices])
        
    new_matrix_n = new_matrix.shape[1]
    if new_matrix_n == len(c):
        new_matrix[-1, :] = np.vstack((new_matrix, c))
    else:
        diff = new_matrix.shape[1] - len(c)
        new_target = np.append(c, np.zeros(diff))
        new_matrix[-1, :] = new_target

    phase_two_matrix = piv_artif(new_matrix, find_basic_columns(new_matrix[:-1, :-1]))
    ind, mat, _ = r_simplex(phase_two_matrix, [])

    if ind:
        opt = -np.round(mat[-1, -1], 8)
        return opt
    else:
        print(f'Phase two simplex no solution.')
        return None
    
    
    
def example1():
    c = [2, 0, 3, 1]
    A = [[0, -1, -1, 1],
         [2, 0, 2, 4],
         [1, 1, 2, 1]]

    b = [3, 12, 3]

    mp = two_phase_simplex_solver(c, A, b)
    print(mp)
    sp = linprog(c, A_eq=A, b_eq=b)
    print(sp['message'])
    print(sp['fun'])



def example2():
    c = [-3, -1, -4, 0, 0]
    A = [[1, 3, 1, 1, 0],
         [3, 1, -1, -1, 0],
         [3, 1, 3, 1, 0],
         [1, 0, 0, 0, 1]]

    b = [10, 2, 6, 1]

    mp = two_phase_simplex_solver(c, A, b)
    print(mp)
    sp = linprog(c, A_eq=A, b_eq=b)
    print(sp['message'])
    print(sp['fun'])

def example3():
    c = [3, 1, 4, 0, 0, 0, 0]
    A = [[1, 3, 1, 1, 0, 0, 0],
         [-3, -1, 1, 0, 1, 0, 0],
         [3, 1, 3, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 0, 1]]

    b = [10, -2, 6, 1]

    mp = two_phase_simplex_solver(c, A, b)
    print(mp)
    sp = linprog(c, A_eq=A, b_eq=b)
    print(sp['message'])
    print(sp['fun'])


if __name__ == '__main__':
    example2()


