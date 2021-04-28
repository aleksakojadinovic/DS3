import numpy as np
from simplex import canonical_simplex as revised_simplex

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
            print(f'\t\t Column {column} is basic.')
        else:
            print(f'\t\t Column {column} is NOT basic.')


    
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

# Construct sub-problem matrix by adding artificial
# objective funcion and do the pivoting thingy
def get_sub_simplex_matrix(subA, eqb, artificial_indices):
    m, n = subA.shape
    num_artif = len(artificial_indices)
    # m + 1 for the new objective function
    # n + 1 for the b
    sub_simplex_matrix = np.zeros((m + 1, n + 1))
    sub_simplex_matrix[:m, :n] = subA
    sub_simplex_matrix[-1, n - num_artif:-1] = np.ones(num_artif)
    sub_simplex_matrix[:-1, -1] = eqb

    for i, artifical_idx in enumerate(artificial_indices):
        sub_simplex_matrix[-1, :] -= sub_simplex_matrix[i, :]

    return sub_simplex_matrix, artificial_indices, [i for i in range(n) if i not in artificial_indices]

# Assumes all constrains have been converted to equalities
def two_phase_simplex_solver(c, eqA, eqb):
    c   = np.array(c, dtype=FLOAT_T)
    eqA = np.array(eqA, dtype=FLOAT_T)
    eqb = np.array(eqb, dtype=FLOAT_T)

    n = len(c)
    if n != eqA.shape[1]:
        raise ValueError(f'Length of objective function is {n} but width of eqA is {eq.shape[1]}')

    m = eqA.shape[0]
    if m != len(eqb):
        raise ValueError(f'eqA has {m} constraints but b-vector has {len(eqb)} values')

    # print('Two phase solver started with')
    # print_constraints(eqA, eqb, ['=' for _ in range(m)], index_from_zero=True)

    subA, artificial_indices = simple_prep(eqA)
    # print('After prep:')
    # print_constraints(subA, eqb, ['=' for _ in range(m)], index_from_zero=True)
    # print(f'Artificial: {artificial_indices}')

    subA, basic_indices, non_basic_indices = get_sub_simplex_matrix(subA, eqb, artificial_indices)
    # print(f'For phase 1: ')
    # print(subA)
    # print(f'where basic: {basic_indices}')
    # print(f'and non basic: {non_basic_indices}')
    sub_m, sub_n = subA.shape
    sub_b_vector = subA[:-1, -1]
    x0length = sub_n - 1
    x0 = np.append(np.zeros(x0length - len(sub_b_vector)), sub_b_vector)

    vector, value, final_basic_indices, final_nonbasic_indices, niter = revised_simplex(subA, np.array(non_basic_indices), np.array(basic_indices), x0)
    print(f'Vector: {vector}')
    print(f'Value: {value}')
    print(f'Basic: {final_basic_indices}')
    print(f'Nonbasic: {final_nonbasic_indices}')
    print(f'Niter: {niter}')
    
    
def example1():
    c = [2, 0, 3, 1]
    A = [[0, -1, -1, 1],
         [2, 0, 2, 4],
         [1, 1, 2, 1]]

    b = [3, 12, 3]

    two_phase_simplex_solver(c, A, b)


def example2():
    c = [-3, -1, -4, 0, 0]
    A = [[1, 3, 1, 1, 0],
         [3, 1, -1, -1, 0],
         [3, 1, 3, 1, 0],
         [1, 0, 0, 0, 1]]

    b = [10, 2, 6, 1]

    two_phase_simplex_solver(c, A, b)

if __name__ == '__main__':
    example1()
