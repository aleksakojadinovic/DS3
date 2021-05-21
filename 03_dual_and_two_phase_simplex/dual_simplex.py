import sys
sys.path.append('../_simplex_utils/')

import numpy as np
import pandas as pd


import lp_parse as lpp
import lp_utils as lpu

import argparse

DEC = 8
FLOAT_T = 'float64'

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

def check_dual_simplex_conditions(matrix):
    c = matrix[-1, :-1]
    if not (c >= 0).all():
        return False, 'Negative c-coeffs found.'
    b = matrix[:-1, -1]
    if (b >= 0).all():
        return False, 'Negative bs found.'

    return True, ''
    
    

"""
f - Target function c0, ... c_{n-1}
A - m x n constraint matrix
b - right side vector
"""
def dual_simplex_(simplex_matrix, *args, **kwargs):
    simplex_matrix = np.array(simplex_matrix, dtype=FLOAT_T)

    if 'basic_column_indices' in kwargs:
        basic_column_indices = kwargs['basic_column_indices']
    else:
        basic_column_indices, _ = find_basic(simplex_matrix[:-1, :-1])


    # print(f'Initial basics are {basic_column_indices}')
    # print(f'in matrix:')
    # print(pd.DataFrame(simplex_matrix))

    canon_m, canon_n = simplex_matrix.shape

    m = canon_m - 1
    n = canon_n - 1

    if len(basic_column_indices) != m:
        raise ValueError(f'No basic submatrix found!')

    initial_conds_met, message = check_dual_simplex_conditions(simplex_matrix)
    if not initial_conds_met:
        print(f'WARNING: Dual simplex condition not met: {message}')
    
    k = 0
    while True:
        A = simplex_matrix[:-1, :-1]
        b = simplex_matrix[:-1, -1]
        c = simplex_matrix[-1, :-1]
        
        if (b >= 0).all():
            result = dict()
            result['bounded'] = True
            result['message'] = 'Optimal value successfully found.'
            result['opt_val'] = -simplex_matrix[-1, -1]
            result['opt_point'] = lpu.fetch_sol_from_simplex_matrix(simplex_matrix, basic_column_indices)
            result['last_matrix'] = simplex_matrix
            result['basic_indices'] = basic_column_indices
            return result
        
        neg_indices = np.argwhere(b < 0).T[0]

        if np.apply_along_axis(np.all, 1, A[neg_indices]).any():
            result = dict()
            result['bounded'] = False
            result['message'] = 'Cond1 failed.'
            result['opt_val'] = -simplex_matrix[-1, -1]
            result['opt_point'] = np.zeros(n)
            result['last_matrix'] = simplex_matrix
            result['basic_indices'] = basic_column_indices
            return result

        s = neg_indices[-1]
        # Now find 
        # max over r of {cr / A_sr such that Asr > 0}
        r = None
        r_max_val = None
        for r_cand in range(n):
            if simplex_matrix[s][r_cand] >= 0:
                continue
            val = c[r_cand]/simplex_matrix[s][r_cand]
            if r is None or val > r_max_val:
                r = r_cand
                r_max_val = val

        if r is None:
            result = dict()
            result['bounded'] = False
            result['message'] = 'R not found.'
            result['opt_val'] = -simplex_matrix[-1, -1]
            result['opt_point'] = np.zeros(n)
            result['last_matrix'] = simplex_matrix
            result['basic_indices'] = basic_column_indices
            return result

        # print(f'Column {r} shall enter current basis {basic_column_indices}')
        # print(f'in matrix')
        # print(pd.DataFrame(simplex_matrix))

        basic_column_indices = lpu.swap_basis(simplex_matrix, basic_column_indices, s, r)
        # print(f'And the new basis is: {basic_column_indices}')
        # print(f'in matrix')
        # print(pd.DataFrame(simplex_matrix))

        for i in range(canon_m):
            if i == s or simplex_matrix[i][r] == 0:
                continue
            coeff = -simplex_matrix[i][r] / simplex_matrix[s][r]
            new_row = simplex_matrix[s] * coeff + simplex_matrix[i]
            simplex_matrix[i] = new_row

        simplex_matrix[s] = simplex_matrix[s] / simplex_matrix[s][r]
        k = k + 1

        
    
        

def eq_to_sm(c, eqA, eqb):
    m, n = eqA.shape
    sm = np.zeros((m + 1, n + 1))
    sm[:m, :n] = eqA
    sm[-1, :-1] = c
    sm[:-1, -1] = eqb
    return sm


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

    res = dual_simplex_(eq_to_sm(c, eqA, eqb))

    if maxx:
        res['opt_val'] *= -1

    res["opt_point"] = res["opt_point"][:orig_n]

    return res



if __name__ == '__main__':
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

    res = solve_lp(c, eqA, eqb, leqA, leqb, args.maximize)


    print(f'>> {res["message"]}')
    if not res['bounded']:
        sys.exit(0)
    
    print(f'Optimal value: ')
    print(f'\t{res["opt_val"]}')
    print(f'Optimum reached for point: ')
    print(f'\t{tuple(res["opt_point"])}')

    # print('----debug')
    # print(f'just to check, dotting with target function: ')
    # print(f'\t\t{np.dot(c, res["opt_point"])}')

