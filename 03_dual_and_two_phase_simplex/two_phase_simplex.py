import sys

from pandas.core.reshape.pivot import pivot
sys.path.append('../_simplex_utils/')

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import warnings

from tableau_simplex import tableau_simplex as t_simplex
from dual_simplex import dual_simplex_ as d_simplex
from lp_utils import find_basic_columns
from lp_utils import sign_zero
from lp_utils import check_for_duality_abc
from lp_utils import pack_to_matrix
import lp_parse as lpp
 

FLOAT_T = 'float64'


UTIL_SIMPLEX_FUNC_ = t_simplex

def adv_prep(eqA, eqb):
    eqA = np.array(eqA, dtype=FLOAT_T)
    eqb = np.array(eqb, dtype=FLOAT_T)
    
    m, n = eqA.shape

    unit_column_indices, unit_column_row_indices = find_basic_columns(eqA, eqb)
    # ----------------------- TMP
    # unit_column_indices, unit_column_row_indices = find_basic(eqA)

    if len(unit_column_indices) == m:
        return eqA, eqb, unit_column_indices, [], []

    for row_idx, col_idx in zip(unit_column_row_indices, unit_column_indices):
        coeff = eqA[row_idx, col_idx]
        eqA[row_idx] /= coeff
        eqb[row_idx] /= coeff


    artificial_unit_row_indices        = [i for i in range(m) if i not in unit_column_row_indices]
    artificial_unit_column_indices     = []
    for missing_row_index in artificial_unit_row_indices:
        artificial_column = np.zeros(m)
        artificial_column[missing_row_index] = 1.0 * sign_zero(eqb[missing_row_index])
        eqA = np.append(eqA, artificial_column.reshape(-1, 1), axis=1)
        artificial_unit_column_indices.append(eqA.shape[1] - 1)


    return eqA, eqb, unit_column_indices + artificial_unit_column_indices, artificial_unit_column_indices, artificial_unit_row_indices


def construct_sub_simplex_matrix(eqA, eqb, artificial_row_indices):
    m, n = eqA.shape
    num_artif = len(artificial_row_indices)
    sub_simplex_matrix = np.zeros((m + 1, n + 1))
    sub_simplex_matrix[:m, :n] = eqA
    sub_simplex_matrix[:-1, -1] = eqb
    sub_simplex_matrix[-1, n - num_artif:-1] = np.ones(num_artif)  

    return sub_simplex_matrix

def pivot_ones(simplex_matrix, row_indices):
    for row_idx in row_indices:
        simplex_matrix[-1, :] -= sign_zero(simplex_matrix[row_idx, -1]) * simplex_matrix[row_idx, :]

    return simplex_matrix

def pivot_coeffs(simplex_matrix, column_indices, row_indices):
    last_row = simplex_matrix[-1, :]
    for col_idx, row_idx in zip(column_indices, row_indices):
        other_row = simplex_matrix[row_idx, :]
        other_row_coeff = other_row[col_idx]
        last_row_coeff = last_row[col_idx]
        mul_coeff = -last_row_coeff / other_row_coeff
        last_row += mul_coeff * other_row
    
    simplex_matrix[-1, :] = last_row
    return simplex_matrix
        

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

def convert_b_to_pos(eqA, eqb):
    for i in range(eqA.shape[0]):
        if eqb[i] < 0:
            eqA[i, :] *= -1
            eqb[i] *= -1

    return eqA, eqb

# ----------------------- TMP
def find_basic(matrix):
    m, n = matrix.shape
    unit_column_indices     = []
    unit_column_row_indices = []
    for j in range(n):
        col = matrix[:, j]
        if len(col[np.isclose(col, 0)]) == m - 1:
            at_row = np.argwhere(np.invert(np.isclose(col, 0)))[0][0]
            if at_row in unit_column_row_indices:
                continue
            unit_column_indices.append(j)
            unit_column_row_indices.append(at_row)

    return unit_column_indices, unit_column_row_indices

def pivot_around(matrix, i0, j0):
    base_coeff = matrix[i0, j0]
    if np.isclose(base_coeff, 0.0):
        raise ValueError(f'Cannot pivot around 0-value')
    base_row   = matrix[i0]
    for i, row in enumerate(matrix):
        if i == i0:
            continue
        if np.isclose(row[j0], 0.0):
            continue
        other_coeff = matrix[i, j0]
        other_row   = matrix[i]
        matrix[i] = (-base_coeff/other_coeff) * base_row + other_row

    return matrix

def phase_one_cleanup(matrix, basic_indices, artificial_indices):
    # STEP 1: Non-basic artificial columns can just be removed
    print(f'Starting matrix cleanup')
    print(f'Matrix: ')
    print(pd.DataFrame(matrix))
    print(f'\tbasic indices: {basic_indices}')
    print(f'\tartif indices: {artificial_indices}')
    m, n = matrix.shape
    cols_to_delete = []
    for j in range(n - 1):
        if j not in basic_indices and j in artificial_indices:
            print(f'\t\tColumn {j} is artificial and not basic so we just schedule it for removal.')
            cols_to_delete.append(j)

    print(f'After removing columns {cols_to_delete} we have')
    artificial_indices = list(filter(lambda x: x not in cols_to_delete, artificial_indices))
    matrix, [basic_indices, artificial_indices] = remove_columns_and_fix_index_lists(matrix, cols_to_delete, [basic_indices, artificial_indices])

    print(f'Matrix: ')
    print(pd.DataFrame(matrix))
    print(f'\tbasic indices: {basic_indices}')
    print(f'\tartif indices: {artificial_indices}')

    if not artificial_indices:
        print(f'No more artificial, so cleanup is done!')
        return matrix

    # STEP 2: Now we have some artificial columns that are indeed basic
    # We distinguish two cases here
    # Step 2a: In this case we remove both row and column entirely
    m, n = matrix.shape
    cols_to_delete = []
    for j in artificial_indices:
        row_idx = np.argwhere(np.invert(np.isclose(matrix[:-1, j], 0.0)))[0][0]
        print(f'\t\tChecking column {matrix[:-1, j]} with its row {matrix[row_idx, :-1]}')
        if len(np.isclose(matrix[row_idx, :-1], 0.0)) == len(matrix[row_idx, :-1] - 1):
            print(f'\t\t\tArtificial and basic column {j} corresponds to row {row_idx} which is all zeros, so remove row and schedule column for removal.')
            cols_to_delete.append(j)
            # We can remove the row immediately
            matrix = np.delete(matrix, obj=row_idx, axis=0)

    print(f'After removing columns {cols_to_delete} we have and their respective rows, we have')
    artificial_indices = list(filter(lambda x: x not in cols_to_delete, artificial_indices))
    matrix, [basic_indices, artificial_indices] = remove_columns_and_fix_index_lists(matrix, cols_to_delete, [basic_indices, artificial_indices])

    print(f'Matrix: ')
    print(pd.DataFrame(matrix))
    print(f'\tbasic indices: {basic_indices}')
    print(f'\tartif indices: {artificial_indices}')

    if not artificial_indices:
        print(f'No more artificial, so cleanup is done!')
        return matrix

    # Step 2b: Some basic and artificial columns have non zero in their respective row
    #          Here we pivot around that value
    for j in artificial_indices:
        print(f'\t\tWe have to pivot around column {j}')
        row_idx = np.argwhere(np.invert(np.isclose(matrix[:-1, j], 0.0)))[0][0]
        row = matrix[row_idx, :-1]
        non_zero_idx = np.argwhere(np.invert(np.isclose(row, 0.0)))[0][0]
        matrix = pivot_around(matrix, non_zero_idx, j)

    return matrix


# Assumes all constrains have been converted to equalities
def two_phase_simplex_solver(c, eqA, eqb, no_dual=False):

    

    c   = np.array(c, dtype=FLOAT_T)
    eqA = np.array(eqA, dtype=FLOAT_T)
    eqb = np.array(eqb, dtype=FLOAT_T)

    n = len(c)
    if n != eqA.shape[1]:
        raise ValueError(f'Length of objective function is {n} but width of eqA is {eqA.shape[1]}')

    m = eqA.shape[0]
    if m != len(eqb):
        raise ValueError(f'eqA has {m} constraints but b-vector has {len(eqb)} values')


    print(f'Starting two phase simplex solver with')
    print(f'A = ')
    print(pd.DataFrame(eqA))
    print(f'b = ')
    print(eqb)

    if not no_dual:
        is_dual, dual_basic_cols, _ = check_for_duality_abc(c, eqA, eqb)
        if is_dual:
            print(f'Dual simplex table detected')
            
            dual_simplex_matrix = pack_to_matrix(eqA, eqb, c)
            print(pd.DataFrame(dual_simplex_matrix))
            return d_simplex(dual_simplex_matrix, basic_column_indices=dual_basic_cols)

    eqA, eqb = convert_b_to_pos(eqA, eqb)
    print(f'Converting all bs to positive, resulting in: ')
    print(f'A = ')
    print(pd.DataFrame(eqA))
    print(f'b = ')
    print(eqb)




    print(f'Starting preparation')
    eqA, eqb, basic_indices, artificial_indices, artificial_row_indices = adv_prep(eqA, eqb)

    print(f'After preparation we have:')
    print(f'A = ')
    print(pd.DataFrame(eqA))
    print(f'b = ')
    print(eqb)
    print(f'Basic indices: {basic_indices}')
    print(f'Artificial indices: {artificial_indices}')
    print(f'Artificial row indices: {artificial_row_indices}')

    sub_problem_simplex_matrix = construct_sub_simplex_matrix(eqA, eqb, artificial_row_indices)

    print(f'We will now append the sub-problem objective function:')
    print(pd.DataFrame(sub_problem_simplex_matrix))

    # sub_problem_simplex_matrix = pivot_coeffs(sub_problem_simplex_matrix, artificial_indices, artificial_row_indices)
    sub_problem_simplex_matrix = pivot_ones(sub_problem_simplex_matrix, artificial_row_indices)

    print(f'Now we eliminate artificial variables from objective function:')
    print(pd.DataFrame(sub_problem_simplex_matrix))

    print(f'Now we send that to phase one simplex, along with basic indices being: ')
    print(basic_indices)

    phase_one_simplex_result = UTIL_SIMPLEX_FUNC_(sub_problem_simplex_matrix, basic_indices, phase=1)

    if not phase_one_simplex_result['bounded']:
        print(f'Phase one simplex is not bounded, its last matrix is: ')
        print(pd.DataFrame(phase_one_simplex_result["last_matrix"]))
        print(f'And its basics are {phase_one_simplex_result["basic_indices"]}')
        return phase_one_simplex_result

    

    last_matrix = phase_one_simplex_result['last_matrix']
    last_basic_indices = phase_one_simplex_result['basic_indices']

    phase_one_opt = phase_one_simplex_result['opt_val']

    if not np.isclose(phase_one_opt, 0.0):
        print(f'Phase one opt is not zero, its last matrix is: ')
        print(pd.DataFrame(phase_one_simplex_result["last_matrix"]))
        print(f'And its basics are {phase_one_simplex_result["basic_indices"]}')
        phase_one_simplex_result['bounded'] = False
        phase_one_simplex_result['message'] = 'Phase one optimum iz non-zero'
        return phase_one_simplex_result


    print(f'Phase one simplex is good, and it gives us this tableau: ')
    print(pd.DataFrame(last_matrix))
    print(f'With last basic indices being: {last_basic_indices}')

    print(f'Now we send it to cleanup')
    new_matrix = phase_one_cleanup(last_matrix, last_basic_indices, artificial_indices)
    print(f'After cleanup our matrix is:')
    print(pd.DataFrame(new_matrix))

    
    print(f'Now we just append our old target function')
    new_matrix_n = new_matrix.shape[1]
    if new_matrix_n == len(c):
        new_matrix[-1, :] = c
    else:
        diff = new_matrix.shape[1] - len(c)
        new_target = np.append(c, np.zeros(diff))
        new_matrix[-1, :] = new_target

    print(pd.DataFrame(new_matrix))
    bbi, bbir = find_basic_columns(new_matrix[:-1, :-1], new_matrix[:-1, -1])

    print(f'This matrix has basic columns: {bbi} and their rows {bbir}')

    print(f'We eliminate basic variables from objective function:')
    phase_two_matrix = pivot_coeffs(new_matrix, bbi, bbir)
    print(pd.DataFrame(phase_two_matrix))

    print(f'And off to simplex it goes!')
    
    phase_two_simplex_result = UTIL_SIMPLEX_FUNC_(phase_two_matrix, bbi, phase=2)
    phase_two_simplex_result['opt_point_phase_two'] = phase_two_simplex_result['opt_point']
    phase_two_simplex_result['opt_point'] = phase_two_simplex_result['opt_point'][:len(c)]

    if not phase_two_simplex_result['bounded']:
        print(f'Phase two simplex is not bounded, its last matrix is: ')
        print(pd.DataFrame(phase_two_simplex_result["last_matrix"]))
        print(f'And its basics are {phase_two_simplex_result["basic_indices"]}')

    print(f'Simplex last tableau: ')
    print(pd.DataFrame(phase_two_simplex_result['last_matrix']))
    print(f'Basics being: {phase_two_simplex_result["basic_indices"]}')
    print(f'And solution: ')
    print(phase_two_simplex_result['opt_point'])


    return phase_two_simplex_result


def solve_lp(c, eqA, eqb, leqA, leqb, maxx=False):
    c = np.array(c)
    eqA = np.array(eqA)
    eqb = np.array(eqb)
    leqA = np.array(leqA)
    leqb = np.array(leqb)
    orig_n = len(c)

    eqA, eqb = lpp.convert_to_eq(leqA, leqb, eqA, eqb)
    c = lpp.pad_right(c, eqA.shape[1])

    if maxx:
        c *= -1
    stdout_backup = sys.stdout
    sys.stdout = open('two_phase_log.txt', 'w')
    res = two_phase_simplex_solver(c, eqA, eqb)
    sys.stdout = stdout_backup

    if maxx:
        res['opt_val'] *= -1

    res["opt_point_ut"] = res["opt_point"]
    res["opt_point"] = res["opt_point"][:orig_n]

    return res


def test_against_scipy(A, b, c):

    warnings.filterwarnings("ignore")
    print(f'Me: ')
    try:
        mr = two_phase_simplex_solver(c, A, b)
        if mr['bounded']:
            print(f'Optimal value: {mr["opt_val"]} reached with x={np.around(mr["opt_point"], 5)}')
            print(f'Just to check, dotting opt point and target: ')
            print(f'\t\t {np.dot(mr["opt_point"], c)}')
        else:
            print(mr)
    except:
        print('ne radi :D')

    print('===========')
    
    print(f'Scipy: ')
    sp = linprog(c, A_eq=A, b_eq=b, method='simplex')
    print(f'Optimal value: {sp["fun"]} reached with x={sp["x"]}')
    
    print(f'BTW TARGET FUNCTION: {c}')
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

def example4():
    c = [6, 3, 0, 0, 0]
    A = [[1, 1, -1, 0, 0],
         [2, -1, 0, -1, 0],
         [0, 3, 0, 0, 1]]
    b = [1, 1, 2]
    test_against_scipy(A, b, c)

def example5():
    c = [2, 3, 1, 0, 0, 0]
    A = [[1, 1, 1, 1, 0, 0],
         [-2, -1, 1, 0, 1, 0],
         [0, 1, -1, 0, 0, 1]]
    b = [40, -10, -10]
    test_against_scipy(A, b, c)

def example6():
    c = [1, 1, -1, -1]
    A = [[1, 2, 3, 1],
         [2, -1, -1, -3]]
    b = [7, -1]
    test_against_scipy(A, b, c)

def example7():
    c = [-2, -3, -4, 0, 0, 0]
    A = [[3, 2, 1, 1, 0, 0],
         [2, 3, 3, 0, 1, 0],
         [1, 1, -1, 0, 0, 1]]
    b = [10, 15, 4]
    test_against_scipy(A, b, c)

if __name__ == '__main__':
    # example1()
    # example2()
    # example3()
    # example4()
    # example5()
    # example6()
    # example7()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', 
                        '--input',
                         help='The input file.',
                         required=True)

    parser.add_argument('-m',
                        '--maximize',
                        action='store_true',
                        help='Maximize the objective function (minimization is default).')

    parser.add_argument('-d',
                        '--debug',
                        action='store_true',
                        help='Debug version (compare results with scipy).')

    args = parser.parse_args()
    input_lines = lpp.read_lines_ds(args.input)

    c, eqA, eqb, leqA, leqb = lpp.parse_any_lp_input(input_lines)

    stdout_backup = sys.stdout
    sys.stdout = open('dummy_out.txt', 'w')
    res = solve_lp(c, eqA, eqb, leqA, leqb, args.maximize)
    sys.stdout = stdout_backup

    print(f'>> {res["message"]}')
    if not res['bounded']:
        sys.exit(0)
    
    print(f'Optimal value: ')
    print(f'\t{res["opt_val"]}')
    print(f'Optimum reached for point: ')
    print(f'\t{tuple(res["opt_point"])}')

    if args.debug:
        x = res["opt_point"]
        print(f'Debug: Checking whether function value is correct: ')
        print(f'\t{tuple(c)} o {tuple(res["opt_point"])} = {np.dot(c, res["opt_point"])}')
        print(f'Debug: Checking constraints:')
        
        for lhs, rhs in zip(eqA, eqb):
            print(f'Constraint: {lhs} = {rhs}')
            print(f'\t{np.dot(lhs, x)} = {rhs}')

        for lhs, rhs in zip(leqA, leqb):
            print(f'Constraint: {lhs} <= {rhs}')
            print(f'\t{np.dot(lhs, x)} <= {rhs}')
    



