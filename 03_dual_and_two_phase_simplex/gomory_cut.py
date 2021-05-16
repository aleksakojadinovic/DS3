import numpy as np
import pandas as pd

from two_phase_simplex import solve_lp
from two_phase_simplex import two_phase_simplex_solver

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

def gomorys_cut(c, eqA, eqb):
    c = np.array(c, dtype=FLOAT_T)
    eqA = np.array(eqA, dtype=FLOAT_T)
    eqb = np.array(eqb, dtype=FLOAT_T)

    original_c = c.copy()

    iteration = 1

    

    while True:
        print(f'***********************************************************************************iteration {iteration}')
        print(f'>> Gomory cut iteration with objective function: {c}')
        print(f'A = ')
        print(pd.DataFrame(eqA))
        print(f'b = ')
        print(eqb)
        tf_simplex_result = two_phase_simplex_solver(c, eqA, eqb)
        
        if not tf_simplex_result['bounded']:
            print(nice_data_frame(tf_simplex_result["last_matrix"]))
            print(f'No solution.')
            break

        current_sol = np.around(tf_simplex_result["opt_point"], 8)
        print(f'Simplex gives solution: {current_sol}')
        
        if all(map(lambda x: x.is_integer(), current_sol)):
            print(f'This solution is integer, done.')
            print(f'Optimal value: {np.dot(current_sol[:len(original_c)], original_c)}')
            break
        print(f'This solution is not integer.')
        
        last_simplex_matrix = tf_simplex_result["last_matrix"]
        print(f'Last matrix:')
        print(nice_data_frame(last_simplex_matrix))

        m, n = last_simplex_matrix.shape
        eqA = last_simplex_matrix[:-1, :-1]
        eqb = last_simplex_matrix[:-1, -1]
        c   = last_simplex_matrix[-1, :-1]
        F   = last_simplex_matrix[-1, -1]

        # C <= F actually becomes one constraint
        


        b_parts             = map(smash, eqb)
        b_fracs             = map(lambda x: x[1], b_parts)
        b_fracs_enum        = enumerate(b_fracs)
        b_best              = max(b_fracs_enum, key=lambda x: x[1])
        b_best_idx          = b_best[0]
        

        cut_lhs = eqA[b_best_idx]
        cut_rhs = eqb[b_best_idx]

        cut_lhs, cut_rhs = construct_cut(cut_lhs, cut_rhs)
        print(f'Now we need to add a cut: {cut_lhs} <= {cut_rhs}')


        new_cut_constr_lhs = np.append(cut_lhs, 1)
        new_cut_constr_rhs = cut_rhs

        new_objective_lhs = np.append(c, 0)

        print(f'Cut constraint (with added slack = 1): {new_cut_constr_lhs} = {new_cut_constr_rhs}')
        print(f'New objective function (old but extended with zeros): {new_objective_lhs}')

        eqA = np.append(eqA, np.zeros(m - 1).reshape(-1, 1), axis=1)

        # We also add c <= F


        eqA = np.vstack((eqA, new_cut_constr_lhs))
        eqb = np.append(eqb, new_cut_constr_rhs)

        c = new_objective_lhs

        if iteration > 5:
            break

        iteration += 1

    
    print(f'iter= {iteration}')

        
def example1():
    c = [-1, -1, 0, 0, 0]
    A = [[-8, 7, 1, 0, 0],
         [1, 6, 0, 1, 0],
         [7, -3, 0, 0, 1]]
    b = [14, 60, 16]
    gomorys_cut(c, A, b)

def example2():
    c = [-1, -1, 0, 0]
    A = [[2, -2, -3, 2],
         [0, 3, 3, -1]]
    b = [5, 3]
    gomorys_cut(c, A, b)


if __name__ == '__main__':
    example1()