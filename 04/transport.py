import numpy as np

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

# TODO: Check if this can be refactored
def argmin_exclude(A, ex_rows, ex_columns):
    min_i = None
    min_j = None
    min_val = None

    for i, row in enumerate(A):
        for j, element in enumerate(row):
            if i in ex_rows or j in ex_columns:
                continue
            if min_val is None or element < min_val:
                min_i = i
                min_j = j
                min_val = element

    return min_i, min_j

# Min cost method
def min_prices_method(C, a, b):
    C = np.array(C, dtype='float64')
    a = np.array(a, dtype='float64')
    b = np.array(b, dtype='float32')
    X = np.zeros(C.shape)
    cap_mask = np.zeros(C.shape)
    m, n = C.shape

    removed_rows = np.array([], dtype='int')
    removed_columns = np.array([], dtype='int')
    
    for iteration in range(m + n - 1):
        p, q = argmin_exclude(C, removed_rows, removed_columns)
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

def potential_method(C, a, b, basis_solution, caps):
    print(f'>> Potential method')
    

def problem_matrix_to_cab(problem_matrix):
    problem_matrix = np.array(problem_matrix)
    C = problem_matrix[:-1, :-1]
    a = problem_matrix[:-1, -1]
    b = problem_matrix[-1, :-1]
    return C, a, b


def example1():
    mat      = [[20, 11, 15, 13, 2],
                [17, 14, 12, 13, 6],
                [15, 12, 18, 18, 7],
                [3, 3, 4, 5, 0]]

    C, a, b = problem_matrix_to_cab(mat)

    
    X, cap_mask = min_prices_method(C, a, b)
    potential_method(C, a, b, X, cap_mask)

if __name__ == '__main__':
    example1()



        
    
    
    

