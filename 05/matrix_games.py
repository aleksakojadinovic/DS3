import numpy as np
from scipy.optimize import linprog
import arg_parsing
import sys

DEC = 7

# TODO: Refactor with list indexing
def reconstruct_vector(X, original_length, anulled_indices):
    new_vec = np.zeros(original_length)
    idx = 0
    for i in range(original_length):
        if i not in anulled_indices:
            new_vec[i] = X[idx]
            idx += 1
        
    return new_vec

def get_partial_matrix(matrix, removed_rows, removed_columns):
    m, n = matrix.shape
    # print(f'Asking for partial matrix of dims {m, n} with rr={removed_rows}, rc={removed_columns}')

    include_rows =      [i for i in range(m) if i not in removed_rows]
    include_columns =   [j for j in range(n) if j not in removed_columns]
    return matrix[include_rows, :][:, include_columns]

# Returns a new reduced game matrix
# and indices of dominated rows and columns
def dominate(game_matrix):
    game_matrix = np.array(game_matrix)
    m, n = game_matrix.shape
    removed_rows = []
    removed_columns = []
    while True:
        # Dominate rows
    
        changed = False
        for i in range(m):
            stopi = False
            if i in removed_rows:
                continue
            for j in range(i+1, m):
                if stopi:
                    break
                if j in removed_rows:
                    continue
                rowi = game_matrix[i, [e for e in range(n) if e not in removed_columns]]
                rowj = game_matrix[j, [e for e in range(n) if e not in removed_columns]]
                if (rowi >= rowj).all():
                    # print(f'\trow {rowi} dominated {rowj}')
                    # Remove j-th row
                    removed_rows.append(j)
                    changed = True
                elif (rowj >= rowi).all():
                    # print(f'\trow {rowj} dominated {rowi}')
                    # Remove i-th row
                    removed_rows.append(i)
                    changed = True
                    stopi = True

        # Dominate columns
        for r in range(n):
            stopr = False
            if r in removed_columns:
                continue
            for s in range(r+1, n):
                if stopr:
                    break
                if s in removed_columns:
                    continue
                colr = game_matrix[[e for e in range(m) if e not in removed_rows], r]
                cols = game_matrix[[e for e in range(m) if e not in removed_rows], s]
                if (cols <= colr).all():
                    # print(f'\tcol {cols} dominated {colr}')
                    # Remove r-th column
                    removed_columns.append(r)
                    changed = True
                    stopr = True
                elif (colr <= cols).all():
                    # Remove s-th column
                    # print(f'\tcol {colr} dominated {cols}')
                    removed_columns.append(s)
                    changed = True
        
        # print(get_partial_matrix(game_matrix, removed_rows, removed_columns))
        if not changed:
            break
    
    # game_matrix = game_matrix[[i for i in range(m) if i not in removed_rows], [j for j in range(n) if j not in removed_columns]]
    return get_partial_matrix(game_matrix, removed_rows, removed_columns), removed_rows, removed_columns
    # return game_matrix, removed_rows, removed_columns


# Prepares matrix for linear programming solver
# If direction is invalid then min is used
def solve_lp_problem(game_matrix, direction):
    game_matrix = np.array(game_matrix)
    if direction not in ['min', 'max']:
        direction = 'min'

    # The matrix represents regular >= or <= constraints
    # If max then >=, if min then <=
    # Scipy uses <= so if min then we keep the signs,
    # if max then we change the signs

    if direction == 'max':
        game_matrix *= -1

    orig_m, orig_n = game_matrix.shape
    # Constraints:
    #   2 for >= 0 for new variables
    #   orig_n for xi >= 0
    #   1 for sum_xi = 1
    ineq_constrains = np.zeros((orig_m, orig_n + 2))
    m, n = ineq_constrains.shape

    # We add original constrains plus the new vars (-v1 + v2)
    ineq_constrains[:orig_m, :orig_n] = game_matrix
    ineq_constrains[:orig_m, orig_n:] = [[-1, 1] if direction == 'min' else [1, -1] for _ in range(orig_m)]
    ineq_b = np.zeros(ineq_constrains.shape[0])

    eq_constraints = np.array([np.zeros(n)])
    eq_constraints[0, :orig_n] = 1
    eq_b = [1]


    

    target = np.zeros(n)
    target[-1] = -1
    target[-2] = 1
    if direction == 'max':
        target *= -1

    # scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='interior-point', callback=None, options=None, x0=None)
    res = linprog(c=target, A_ub=ineq_constrains, b_ub=ineq_b, A_eq=eq_constraints, b_eq=eq_b)


    actual_fun = res.fun
    actual_x = np.append(res.x[:-2], [res.x[-2] - res.x[-1]])
    return actual_fun if direction == 'min' else -actual_fun, actual_x


    

    
        


    






def solve_game(game_matrix):
    game_matrix = np.array(game_matrix)
    m, n = game_matrix.shape

    game_matrix_dominated, removed_rows, removed_columns = dominate(game_matrix)
    f, y = solve_lp_problem(game_matrix_dominated, 'min')
    _, x = solve_lp_problem(game_matrix_dominated.T, 'max')

    x = reconstruct_vector(x, n, removed_rows)
    y = reconstruct_vector(y, n, removed_columns)
    
    return np.around(f, DEC), np.round(x, DEC), np.round(y, DEC)

    


def example1():
    M = [[2, 1, 2, 3],
        [3, 1.5, 1, 2],
        [2, 2, 1, 1],
        [1, 1, 1, 1]]

    print(solve_game(M))


def example2():
    M = [[1, -1, -1],
        [-1, -1, 3],
        [-1, -2, -1]]

    print(solve_game(M))

def example3():
    M = [[100, 0],
        [-100, 200]]

    print(solve_game(M))



if __name__ == '__main__':
    in_matrix = arg_parsing.read_input()
    if in_matrix is None:
        print(f'Parse error occurred.')
        sys.exit(1)
    
    v, x, y = solve_game(in_matrix)
    print(f'v = {v}')
    print(f'x = {tuple(list(x))}')
    print(f'y = {tuple(list(y))}')