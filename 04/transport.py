import numpy as np
import sys

import graphs
import utils as ut
import arg_parsing as ap

DUMMY_VALUE = 0

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



# Min cost method
def min_cost_method(C, a, b):
    C = np.array(C, dtype='float64')
    a = np.array(a, dtype='float64')
    b = np.array(b, dtype='float32')
    X = np.zeros(C.shape)
    cap_mask = np.zeros(C.shape)
    m, n = C.shape

    removed_rows = np.array([], dtype='int')
    removed_columns = np.array([], dtype='int')
    
    for iteration in range(m + n - 1):
        p, q = ut.argmin_exclude(C, removed_rows, removed_columns)
        the_min = min(a[p], b[q])
        X[p][q] = the_min

        a[p] = a[p] - the_min
        b[q] = b[q] - the_min

        if a[p] == 0:
            removed_rows = np.append(removed_rows, p)
        if b[q] == 0:
            removed_columns = np.append(removed_columns, q)

        cap_mask[p][q] = 1

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
        basis_indices = list(map(tuple, np.argwhere(caps == 1)))
        non_basis_indices = list(map(tuple, np.argwhere(caps == 0)))

        potentials_systemA = np.zeros((m + n - 1, m + n))
        potentials_systemB = np.zeros(m + n - 1)
        for row, base_coords in enumerate(basis_indices):
            base_i = base_coords[0]
            base_j = base_coords[1]
            potentials_systemA[row][base_i] = 1.0
            potentials_systemA[row][m + base_j] = 1.0
            potentials_systemB[row] = C[base_i][base_j]

        to_set_zero_index, to_set_zero_axis = find_best_potential(caps)

        to_set_zero_actual_index = to_set_zero_index
        if to_set_zero_axis == 1:
            to_set_zero_index = m + to_set_zero_index

        potentials_systemA = np.delete(potentials_systemA, to_set_zero_actual_index, 1)

        potential_system_solution = np.linalg.solve(potentials_systemA, potentials_systemB)
        potential_system_solution = np.insert(potential_system_solution, to_set_zero_actual_index, 0)

        r = None
        s = None
        lowest_val = None
        for i, j in non_basis_indices:
            # Cij - ui - vj >= 0
            ui = potential_system_solution[i]
            vj = potential_system_solution[m + j]
            val = C[i][j] - ui - vj
            if val < 0:
                if lowest_val is None or val < lowest_val:
                    lowest_val = val
                    r = i
                    s = j
        
        if lowest_val is None:
            return basis_solution

        graph = graphs.get_graph(r, s, caps)
        cycle = graphs.find_cycle(graph, ut.pack_indices(r, s, shape), shape=shape)
        cycle_coordinates = list(map(lambda x: ut.unpack_index(x, shape), cycle))

        initial_theta = lowest_val

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
        
        # Now X_r_s is supposed to enter the basis
        # And and XijB min leaves the basis
        for idx, (i, j) in enumerate(cycle_coordinates):
            coeff = 1 if idx % 2 == 0 else -1
            basis_solution[i][j] += coeff * corr_theta

        basis_solution[r][s] = corr_theta              

        caps[corr_theta_i][corr_theta_j] = 0
        caps[r][s] = 1
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
    C, a, b, fic_row, fic_col = balance_problem(C, a, b)
    X, cap_mask = min_cost_method(C, a, b)
    basis_matrix = potential_method(C, a, b, X, cap_mask)
    print(interpret_potential_method_results(basis_matrix, C, fic_row, fic_col))

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
    
    just_runnit(problem_matrix)



        
    
    
    

