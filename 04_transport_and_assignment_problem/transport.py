import numpy as np
import sys

import graphs
import utils as ut
import arg_parsing as ap
import pandas as pd

DUMMY_VALUE = 1000

### The transportation problem

# There are `m` shops, indexed with `i`
# and `n` warehouses indexed with `j`
# `c_ij` is the cost of transportation from `i` to `j`
# `x_ij` is the transported amount (target variables)
# `a_i` is the amount of goods at warehouse `i` (SUPPLY)
# `b_j` is the required amount of goods are shop `j` (DEMAND)

# Find optimal x_ij.

### MODEL:
# (min) sum sum c_ij*x_ij
# such that
# sum_{i=0,m-1} x_ij <= a_i  for all i  (DO NOT EXCEED SUPPLY)
# sum_{i=0,n-1} x_ij <= b_j  for all j  (DO NOT EXCEED DEMAND)
# xij >= 0                              (NON NEGATIVE AMOUNTS)

LOG_FILE = '.dummy.txt'

def get_problem_df(C, a, b):
    m, n = C.shape
    problem_matrix_ = np.zeros((m+1, n+1), dtype='int32')
    problem_matrix_[:m, :n] = C
    problem_matrix_[:-1, -1] = a
    problem_matrix_[-1, :-1] = b

    return pd.DataFrame(problem_matrix_,
                        index=[str(i) for i in range(m)] + ['b'],
                        columns=[str(i) for i in range(n)] + ['a'])

# Min cost method
def min_cost_method(C, a, b):
    problem_df_ = get_problem_df(C, a, b)
    print(f' > Starting min cost method for: ')
    print(problem_df_)

    
    C = np.array(C, dtype='float64')
    a = np.array(a, dtype='float64')
    b = np.array(b, dtype='float32')
    X = np.zeros(C.shape)
    cap_mask = np.zeros(C.shape)
    m, n = C.shape

    removed_rows = np.array([], dtype='int')
    removed_columns = np.array([], dtype='int')
    
    for _ in range(m + n - 1):
        p, q = ut.argmin_exclude(C, removed_rows, removed_columns)

        print(f' >> Min position is {p, q} with cost = {C[p][q]}, supply = {a[p]}, demand = {b[q]}')

        the_min = min(a[p], b[q])

        X[p][q] = the_min

        a[p] = a[p] - the_min
        b[q] = b[q] - the_min

        
        if a[p] == 0:
            print(f'\t >> Removing row {p}')
            removed_rows = np.append(removed_rows, p)
        elif b[q] == 0:
            print(f'\t >> Removing column {q}')
            removed_columns = np.append(removed_columns, q)
           

        cap_mask[p][q] = 1
    print(f'> Basic feasible solution found by min cost method: ')
    print(pd.DataFrame(X))
    return X, cap_mask

# Returns index and axis of a potential
# that has the most basic variables in either its
# row or its column
def find_best_potential(caps_mask):
    best_row_index = None
    best_row_count = None
    # Check rows first
    for i, row in enumerate(caps_mask):
        basic_count = np.count_nonzero(row == 1)
        if best_row_index is None or basic_count > best_row_count:
            best_row_index = i
            best_row_count = basic_count
    
    best_column_index = None
    best_column_count = None
    for j in range(caps_mask.shape[1]):
        col = caps_mask[:, j]
        basic_count = np.count_nonzero(col == 1)
        if best_column_index is None or basic_count > best_column_count:
            best_column_index = j
            best_column_count = basic_count

    return (best_row_index, 0) if best_row_count > best_column_count else (best_column_index, 1)


def potential_method(C, a, b, basis_solution, caps):
    m, n = C.shape
    shape = C.shape
    # Now we need to find ui, vj using:
    # ui + vj = cijB
    iteration = 0
    while True:
        print(f'>>> Potential method Iteration {iteration}')
        basis_indices = list(map(tuple, np.argwhere(caps == 1)))
        non_basis_indices = list(map(tuple, np.argwhere(caps == 0)))

        print(f'basis solution: ')
        print(pd.DataFrame(basis_solution))
        print(f'\t basic indices: {basis_indices}')

        potentials_systemA = np.zeros((m + n - 1, m + n))
        potentials_systemB = np.zeros(m + n - 1)
        for row, base_coords in enumerate(basis_indices):
            base_i = base_coords[0]
            base_j = base_coords[1]
            potentials_systemA[row][base_i] = 1.0
            potentials_systemA[row][m + base_j] = 1.0
            potentials_systemB[row] = C[base_i][base_j]

        print(f'System before removal: ')
        print(potentials_systemA)
        print(potentials_systemB)

        to_set_zero_index, to_set_zero_axis = find_best_potential(caps)

        print(f'Choosing to anull potential {"u" if to_set_zero_index == 0 else "v"}_{to_set_zero_index}')

        to_set_zero_actual_index = to_set_zero_index
        if to_set_zero_axis == 1:
            to_set_zero_index = m + to_set_zero_index

        potentials_systemA = np.delete(potentials_systemA, to_set_zero_actual_index, 1)

        print(f'After anulling we have a new system: ')
        print(potentials_systemA)

        potential_system_solution = np.linalg.solve(potentials_systemA, potentials_systemB)
        potential_system_solution = np.insert(potential_system_solution, to_set_zero_actual_index, 0)

        print(f'The system solution is: {potential_system_solution}')

        print(f'Finding correctional start: ')
        r = None
        s = None
        lowest_val = None
        for i, j in non_basis_indices:
            # Cij - ui - vj >= 0
            ui = potential_system_solution[i]
            vj = potential_system_solution[m + j]
            val = C[i][j] - ui - vj
            print(f'\tC_{{{i}{j}}} - u_{i} - v_{j} = {C[i][j]} - {ui} - {vj} = {val}')
            if val < 0:
                if lowest_val is None or val < lowest_val:
                    lowest_val = val
                    r = i
                    s = j
        
        if lowest_val is None:
            return basis_solution

        print(f'Choosing r, s = {r, s}')

        graph = graphs.get_graph(r, s, caps)

        print('Constructed graph: ')
        graphs.nice_print_(graph, (m, n))

        cycle = graphs.find_cycle(graph, ut.pack_indices(r, s, shape), shape=shape)
        cycle_coordinates = list(map(lambda x: ut.unpack_index(x, shape), cycle))

        print(f'Cycle coordinates: {cycle_coordinates}')

        # Correctional theta min XijB where XijB is in the cycle
        corr_theta_i = None
        corr_theta_j = None
        corr_theta = None
        for idx, (i, j) in enumerate(cycle_coordinates):
            if idx % 2 == 1:
                if corr_theta is None or basis_solution[i][j] < corr_theta:
                    corr_theta_i = i
                    corr_theta_j = j
                    corr_theta = basis_solution[i][j]
        
        print(f'Initial correction chosen at position {corr_theta_i, corr_theta_j} with value {corr_theta}')

        # Now X_r_s is supposed to enter the basis
        # And and XijB min leaves the basis
        for idx, (i, j) in enumerate(cycle_coordinates):
            coeff = 1 if idx % 2 == 0 else -1
            print(f'Position {i, j} with value {basis_solution[i][j]} gets {"+" if coeff == 1 else "-"}theta')
            basis_solution[i][j] += coeff * corr_theta

        

        basis_solution[r][s] = corr_theta              

        caps[corr_theta_i][corr_theta_j] = 0
        caps[r][s] = 1

        print(f'New basic solution: ')
        print(basis_solution)

        iteration += 1

def interpret_potential_method_results(solution, C, fictional_rows=[], fictional_columns=[]):
    sol = 0
    for i in range(C.shape[0]):
        if i in fictional_rows:
            continue
        for j in range(C.shape[1]):
            if j in fictional_columns:
                continue
            sol += solution[i][j]*C[i][j]

    return sol
            
def balance_assignment_problem(C, a, b):
    supply = sum(a)
    demand = sum(b)
    if supply == demand:
        return C, a, b, [], []
    
    diff = int(abs(supply - demand))
    frow = []
    fcol = []
    if supply > demand:
        # We need to add `diff` columns
        new_col = np.repeat(DUMMY_VALUE, C.shape[0])
        for _ in range(diff):
            C = np.hstack((C, new_col.reshape(-1, 1)))
            b = np.append(b, 1)
            fcol.append(C.shape[1] - 1)
    else:
        new_row = np.repeat(DUMMY_VALUE, C.shape[1])
        for _ in range(diff):
            C = np.vstack((C, new_row))
            a = np.append(a, 1)
            frow.append(C.shape[0] - 1)

    return C, a, b, frow, fcol

def balance_problem(C, a, b):
    supply = sum(a)
    demand = sum(b)

    if supply == demand:
        return C, a, b, [], []


    diff = abs(supply - demand)
    frow = []
    fcol = []
    if supply > demand:
        # We need to add a column
        new_col = np.repeat(DUMMY_VALUE, C.shape[0])
        C = np.hstack((C, new_col.reshape(-1, 1)))
        b = np.append(b, diff)
        fcol = [C.shape[1] - 1]
    else:
        # We need to add a row
        new_row = np.repeat(DUMMY_VALUE, C.shape[1])
        C = np.vstack((C, new_row))
        a = np.append(a, diff)
        frow = [C.shape[0] - 1]
    
    return C, a, b, frow, fcol
        

def just_runnit(mat):
    C, a, b = ut.problem_matrix_to_cab(mat)
    is_assignment = (a == 1).all() and (b == 1).all()
    balancing_function = balance_assignment_problem if is_assignment else balance_problem
    C, a, b, fic_row, fic_col = balancing_function(C, a, b)
    X, cap_mask = min_cost_method(C, a, b)
    basis_matrix = potential_method(C, a, b, X, cap_mask)
    print(interpret_potential_method_results(basis_matrix, C, fic_row, fic_col))
    actual_num_rows = basis_matrix.shape[0] - len(fic_row)
    actual_num_cols = basis_matrix.shape[1] - len(fic_col)
    actual_basis_matrix = basis_matrix[:actual_num_rows, :actual_num_cols]
    print(actual_basis_matrix)

def example1():
    mat      = [[20, 11, 15, 13, 2],
                [17, 14, 12, 13, 6],
                [15, 12, 18, 18, 7],
                [3, 3, 4, 5, 0]]
    just_runnit(mat)

def example2():
    mat = [[3, 9, 8, 10, 4, 28],
           [6, 10, 3, 2, 3, 13],
           [3, 2, 7, 10, 3, 19],
           [3, 2, 3, 2, 8, 18],
           [24, 16, 10, 20, 22, 0]]

    just_runnit(mat)

    
if __name__ == '__main__':
    problem_matrix = ap.read_input()
    if problem_matrix is None:
        print('Invalid input file.')
        sys.exit(1)
    
    stdout_backup = sys.stdout
    sys.stdout = open(LOG_FILE, 'w')
    just_runnit(problem_matrix)



        
    
    
    

