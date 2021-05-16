import argparse
import numpy as np
import pandas as pd

from two_phase_simplex import two_phase_simplex_solver
import lp_parse as lpp
import sys


def nice_data_frame(simplex_matrix):
    m, n = simplex_matrix.shape
    df = pd.DataFrame(simplex_matrix, columns=[f'x_{i}' for i in range(n-1)] + ['b'])
    return df

def smash(x):
    return np.floor(x), x - np.floor(x)

def construct_cut(lhs, rhs):
    # print(f'Constructing cut for constraint: {lhs} = {rhs}')

    lhs = list(map(smash, lhs))
    rhs = smash(rhs)

    # print(f'Smashed: {lhs} = {rhs}')

    new_lhs = np.array(list(map(lambda x: x[1], lhs)))
    new_rhs = rhs[1]

    # print(f'Cut: {new_lhs} = {new_rhs}')

    return -new_lhs, -new_rhs


FLOAT_T = 'float64'

def gomorys_cutting_plane_solver(c, eqA, eqb):
    c = np.array(c, dtype=FLOAT_T)
    eqA = np.array(eqA, dtype=FLOAT_T)
    eqb = np.array(eqb, dtype=FLOAT_T)

    original_c = c.copy()

    iteration = 1

    while True:
        tf_simplex_result = two_phase_simplex_solver(c, eqA, eqb)
        
        if not tf_simplex_result['bounded']:
            result = dict()
            result['message'] = 'The function is unbounded'
            result['bounded'] = False
            result['iterations'] = iteration
            result['opt_point'] = np.zeros(len(original_c))
            result['opt_val'] = 0.0
            return result

        current_sol = np.around(tf_simplex_result["opt_point"], 8)
        
        if all(map(lambda x: x.is_integer(), current_sol)):
            print(f'This solution is integer, done.')
            print(f'Optimal value: {np.dot(current_sol[:len(original_c)], original_c)}')
            result = dict()
            result['message'] = 'Successfully found optimum value'
            result['bounded'] = True
            result['iterations'] = iteration
            result['opt_point'] = current_sol[:len(original_c)]
            result['opt_val'] = np.dot(current_sol[:len(original_c)], original_c)
            return result
        
        last_simplex_matrix = tf_simplex_result["last_matrix"]

        m, n = last_simplex_matrix.shape
        eqA = last_simplex_matrix[:-1, :-1]
        eqb = last_simplex_matrix[:-1, -1]
        c   = last_simplex_matrix[-1, :-1]

        b_parts             = map(smash, eqb)
        b_fracs             = map(lambda x: x[1], b_parts)
        b_fracs_enum        = enumerate(b_fracs)
        b_best              = max(b_fracs_enum, key=lambda x: x[1])
        b_best_idx          = b_best[0]
        

        cut_lhs = eqA[b_best_idx]
        cut_rhs = eqb[b_best_idx]

        cut_lhs, cut_rhs = construct_cut(cut_lhs, cut_rhs)
        new_cut_constr_lhs = np.append(cut_lhs, 1)
        new_cut_constr_rhs = cut_rhs

        new_objective_lhs = np.append(c, 0)

        eqA = np.append(eqA, np.zeros(m - 1).reshape(-1, 1), axis=1)
        eqA = np.vstack((eqA, new_cut_constr_lhs))
        eqb = np.append(eqb, new_cut_constr_rhs)
        c = new_objective_lhs

        iteration += 1

def solve_ip(c, eqA, eqb, leqA, leqb, maxx=False):
    c = np.array(c)

    orig_n = len(c)

    eqA = np.array(eqA)
    eqb = np.array(eqb)
    leqA = np.array(leqA)
    leqb = np.array(leqb)
    eqA, eqb = lpp.convert_to_eq(leqA, leqb, eqA, eqb)

    c = lpp.pad_right(c, eqA.shape[1])

    

    if maxx:
        c *= -1

    res = gomorys_cutting_plane_solver(c, eqA, eqb)

    if maxx:
        res['opt_val'] *= -1

    res['opt_point'] = res['opt_point'][:orig_n]
    return res



        
def example1():
    c = [-1, -1, 0, 0, 0]
    A = [[-8, 7, 1, 0, 0],
         [1, 6, 0, 1, 0],
         [7, -3, 0, 0, 1]]
    b = [14, 60, 16]
    gomorys_cutting_plane_solver(c, A, b)

def example2():
    c = [-1, -1, 0, 0]
    A = [[2, -2, -3, 2],
         [0, 3, 3, -1]]
    b = [5, 3]
    gomorys_cutting_plane_solver(c, A, b)


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

    stdout_backup = sys.stdout
    sys.stdout = open('dummy_out.txt', 'w')
    res = solve_ip(c, eqA, eqb, leqA, leqb, args.maximize)
    sys.stdout = stdout_backup

    print(f'>> {res["message"]}')
    if not res['bounded']:
        sys.exit(0)
    
    print(f'Optimal value: ')
    print(f'\t{res["opt_val"]}')
    print(f'Optimum reached for point: ')
    print(f'\t{tuple(res["opt_point"])}')