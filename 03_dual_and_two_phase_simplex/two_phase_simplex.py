import sys
sys.path.append('../_simplex_utils/')

import argparse

import numpy as np
from scipy.optimize import linprog
import warnings

from tableau_simplex import tableau_simplex as t_simplex
from lp_utils import find_basic_columns
from lp_utils import sign_zero
import lp_parse as lpp
 

FLOAT_T = 'float64'

def adv_prep(eqA, eqb):
    eqA = np.array(eqA, dtype=FLOAT_T)
    eqb = np.array(eqb, dtype=FLOAT_T)
    
    m, n = eqA.shape

    unit_column_indices, unit_column_row_indices = find_basic_columns(eqA, eqb)

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


    # print(f'Starting two phase simplex solver with')
    # print(f'A = ')
    # print(eqA)
    # print(f'b = ')
    # print(eqb)

    eqA, eqb = convert_b_to_pos(eqA, eqb)
    # print(f'Converting all bs to positive, resulting in: ')
    # print(f'A = ')
    # print(eqA)
    # print(f'b = ')
    # print(eqb)


    # print(f'Starting preparation')
    eqA, eqb, basic_indices, artificial_indices, artificial_row_indices = adv_prep(eqA, eqb)

    # print(f'After preparation we have:')
    # print(f'A = ')
    # print(eqA)
    # print(f'b = ')
    # print(eqb)
    # print(f'Basic indices: {basic_indices}')
    # print(f'Artificial indices: {artificial_indices}')
    # print(f'Artificial row indices: {artificial_row_indices}')

    sub_problem_simplex_matrix = construct_sub_simplex_matrix(eqA, eqb, artificial_row_indices)

    # print(f'We will now append the sub-problem objective function:')
    # print(sub_problem_simplex_matrix)

    sub_problem_simplex_matrix = pivot_ones(sub_problem_simplex_matrix, artificial_row_indices)

    # print(f'Now we do the pivot thingy and get:')
    # print(sub_problem_simplex_matrix)

    # print(f'Now we send that to phase one simplex, along with basic indices being: ')
    # print(basic_indices)

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


    # print(f'Phase one simplex is good, and it gives us this tableau: ')
    # print(pd.DataFrame(last_matrix))
    # print(f'With last basic indices being: {last_basic_indices}')

    cols_to_delete = []
    for artif_index in artificial_indices:
        if artif_index in last_basic_indices:
            continue
        cols_to_delete.append(artif_index)

    artificial_indices = [a for a in artificial_indices if a not in cols_to_delete]

    new_matrix, [artificial_indices] = remove_columns_and_fix_index_lists(last_matrix, cols_to_delete, [artificial_indices])

    cols_to_delete = []
    for art_and_basic in artificial_indices:
        row_idx = np.argwhere(new_matrix[:, art_and_basic] == 1)[0][0]
        row = new_matrix[row_idx]
        
        if (len(row[np.invert(np.isclose(row, 0.0))]) == len(row) - 1):
            artificial_indices.remove(art_and_basic)
            cols_to_delete.append(art_and_basic)
            new_matrix = np.delete(new_matrix, [row_idx], axis=0)

 
    new_matrix, [artificial_indices] = remove_columns_and_fix_index_lists(new_matrix, cols_to_delete, [artificial_indices])
    
    # print(f'After removing all kinds of stuff from it we have the following matrix: ')
    # print(pd.DataFrame(new_matrix))

    
    # print(f'Now we just append our old target function')
    new_matrix_n = new_matrix.shape[1]
    if new_matrix_n == len(c):
        new_matrix[-1, :] = np.vstack((new_matrix, c))
    else:
        diff = new_matrix.shape[1] - len(c)
        new_target = np.append(c, np.zeros(diff))
        new_matrix[-1, :] = new_target

    # print(pd.DataFrame(new_matrix))
    bbi, bbir = find_basic_columns(new_matrix[:-1, :-1], new_matrix[:-1, -1])
    # print(f'This matrix has basic columns: {bbi} and their rows {bbir}')

    # print(f'We perform the pivoting thingy on it')
    phase_two_matrix = pivot_coeffs(new_matrix, bbi, bbir)
    # print(pd.DataFrame(phase_two_matrix))

    # print(f'And off to simplex it goes!')
    
    phase_two_simplex_result = t_simplex(phase_two_matrix, bbi, phase=2)
    phase_two_simplex_result['opt_point_phase_two'] = phase_two_simplex_result['opt_point']
    phase_two_simplex_result['opt_point'] = phase_two_simplex_result['opt_point'][:len(c)]
    # print(f'Simplex last tableau: ')
    # print(pd.DataFrame(phase_two_simplex_result['last_matrix']))
    # print(f'Basics being: {phase_two_simplex_result["basic_indices"]}')
    # print(f'And solution: ')
    # print(phase_two_simplex_result['opt_point'])


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

    res = two_phase_simplex_solver(c, eqA, eqb)

    if maxx:
        res['opt_val'] *= -1

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
    



