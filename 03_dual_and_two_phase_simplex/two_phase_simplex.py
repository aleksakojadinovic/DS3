import numpy as np
from scipy.linalg import basic
from tableau_simplex import tableau_simplex as t_simplex
import lp_parse as lpp
import sys
from scipy.optimize import linprog
import warnings

FLOAT_T = 'float32'

def find_basic_columns(mat):

    m, n = mat.shape
    unit_column_indices     = []
    unit_column_row_indices = []
    for j in range(n-1):
        col = mat[:-1, j]
        if len(col[col == 0]) == m - 2:
            at_row = np.argwhere(col != 0)[0][0]
            if at_row in unit_column_row_indices:
                continue
            unit_column_indices.append(j)
            unit_column_row_indices.append(at_row)

    # print(f'\t\t\t --- {unit_column_indices}')
    return unit_column_indices, unit_column_row_indices


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

            if np.sign(col[at_row]) != np.sign(eqb[at_row]):
                continue
            
            unit_column_indices.append(j)
            unit_column_row_indices.append(at_row)

    if len(unit_column_indices) == m:
        return eqA, eqb, unit_column_indices, [], []

    for row_idx, col_idx in zip(unit_column_row_indices, unit_column_indices):
        coeff = eqA[row_idx, col_idx]
        eqA[row_idx] /= coeff
        eqb[row_idx] /= coeff


    artificial_unit_row_indices        = [i for i in range(m) if i not in unit_column_row_indices]
    artificial_unit_column_indices  = []
    for missing_row_index in artificial_unit_row_indices:
        artificial_column = np.zeros(m)
        artificial_column[missing_row_index] = 1.0 * np.sign(eqb[missing_row_index])
        eqA = np.append(eqA, artificial_column.reshape(-1, 1), axis=1)
        artificial_unit_column_indices.append(eqA.shape[1] - 1)


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

    # print('donig thingy to')
    # print(sub_simplex_matrix)
    


    for ar in artificial_row_indices:
        sub_simplex_matrix[-1, :] -= np.sign(eqb[ar]) * sub_simplex_matrix[ar, :]


    return sub_simplex_matrix

def piv_artif(sub_simplex_matrix, artificial_indices):
    for i, artifical_idx in enumerate(artificial_indices):
        sub_simplex_matrix[-1, :] -= sub_simplex_matrix[i, :]

    return sub_simplex_matrix

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

    
    # print('sending to prep: ')
    # print(eqA)
    # print(eqb)

    eqA, eqb, basic_indices, artificial_indices, artificial_row_indices = adv_prep(eqA, eqb)

    # print('after prep we have:')
    # print(eqA)
    # print(eqb)

    sub_problem_simplex_matrix = get_sub_simplex_matrix(eqA, eqb, artificial_indices, artificial_row_indices)

    # print('sub problem matrix:')
    # print(sub_problem_simplex_matrix)


    phase_one_simplex_result = t_simplex(sub_problem_simplex_matrix, basic_indices, phase=1)

    if not phase_one_simplex_result['bounded']:
        return phase_one_simplex_result

    last_matrix = phase_one_simplex_result['last_matrix']
    last_basic_indices = phase_one_simplex_result['basic_indices']

    phase_one_opt = phase_one_simplex_result['opt_val']

    if not np.isclose(phase_one_opt, 0.0):
        phase_one_simplex_result['bounded'] = False
        phase_one_simplex_result['message'] = 'Phase one optimum iz non-zero'
        return phase_one_simplex_result

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

    bbi, _ = find_basic_columns(new_matrix[:-1, :-1])

    phase_two_matrix = piv_artif(new_matrix, bbi)
    phase_two_basic,_ = find_basic_columns(phase_two_matrix)


    # phase_two_A, phase_two_b, phase_two_basic, _, _  = adv_prep(phase_two_matrix[:-1, :-1], phase_two_matrix[:-1, -1])


    phase_two_simplex_result = t_simplex(phase_two_matrix, phase_two_basic, phase=2)
    return phase_two_simplex_result

    

def test_against_scipy(A, b, c):
    print(f'TARGET FUNCTION: {c}')
    warnings.filterwarnings("ignore")
    print(f'Me: ')
    mr = two_phase_simplex_solver(c, A, b)
    if mr['bounded']:
        print(f'Optimal value: {mr["opt_val"]} reached with x={np.around(mr["opt_point"], 8)}')
    else:
        print(mr)


    print('===========')
    
    print(f'Scipy: ')
    sp = linprog(c, A_eq=A, b_eq=b, method='simplex')
    print(f'Optimal value: {sp["fun"]} reached with x={sp["x"]}')

    print(f'==============================================================================')
    
def example1():
    c = [2, 0, 3, 1]
    A = [[0, -1, -1, 1],
         [2, 0, 2, 4],
         [1, 1, 2, 1]]

    b = [3, 12, 3]

    test_against_scipy(A, b, c)



def example2():
    c = [-3, -1, -4, 0, 0]
    A = [[1, 3, 1, 1, 0],
         [3, 1, -1, -1, 0],
         [3, 1, 3, 1, 0],
         [1, 0, 0, 0, 1]]

    b = [10, 2, 6, 1]

    test_against_scipy(A, b, c)

def example3():
    c = [3, 1, 4, 0, 0, 0, 0]
    A = [[1, 3, 1, 1, 0, 0, 0],
         [-3, -1, 1, 0, 1, 0, 0],
         [3, 1, 3, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 0, 1]]

    b = [10, -2, 6, 1]

    test_against_scipy(A, b, c)

if __name__ == '__main__':
    example1()
    example2()
    example3()
    # input_file = sys.argv[-1]
    # lines = lpp.read_lines_ds(input_file)
    # eqA, eqb, leqA, leqb = lpp.parse_any_lp_input(lines)




