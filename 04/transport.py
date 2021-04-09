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


def construct_graph(theta_i, theta_j, caps):
    m, n = caps.shape
    graph_matrix = [[] for _ in range(m*n)]
    
    for i in range(m):
        for j in range(n):
            if caps[i][j] == 0 and (i, j) != (theta_i, theta_j):
                continue

            # Row edges
            for other_row in range(m):
                if other_row == i or (caps[other_row][j] == 0 and (other_row, j) != (theta_i, theta_j)):
                    continue
                
                # Edge between (i, j) and (other_row, j)
                graph_matrix[i*n + j].append(other_row*n + j)


            # Column edges
            for other_col in range(n):
                if other_col == j or (caps[i][other_col] == 0 and (i, other_col) != (theta_i, theta_j)):
                    continue

                # Edge between (i, j) and (i, other_col)
                graph_matrix[i*n + j].append(i*n + other_col)
                

    return graph_matrix

    

def is_stable_path(path, m, n):

    last_state = None
    current_state = None
    for i in range(1, len(path)):
        x1, y1 = np.unravel_index(path[i-1], (m, n))
        x2, y2 = np.unravel_index(path[i], (m, n))

        if x1 == x2:
            current_state = 1
        elif y1 == y2:
            current_state = 2
        else:
            return False

        if current_state == last_state:
            return False

        last_state = current_state

    x1, y1 = np.unravel_index(path[-1], (m, n))
    x2, y2 = np.unravel_index(path[0], (m, n))

    if x1 == x2:
        current_state = 1
    elif y1 == y2:
        current_state = 2
    else:
        return False

    if current_state == last_state:
        return False

    return True


def get_cycle(graph, start_node, t_m, t_n):

    num_nodes = len(graph)
    parents = [-1 for _ in range(num_nodes)]
    on_stack = [False for _ in range(num_nodes)]

    s = [(start_node, 0)]
    on_stack[start_node] = True
    path = []

    while len(s) > 0:
        node = s[-1][0]
        index = s[-1][1]

        if index == len(graph[node]):
            s = s[:-1]
            on_stack[node] = False
            parents[node] = -1
            continue

        val1 = (s[-1])[0]
        val2 = (s[-1])[1] + 1
        s = s[:-1]
        s.append((val1, val2))

        child = graph[node][index]

        if child == start_node and parents[node] != start_node:

            if len(s) % 2 == 0:
                path.append(start_node)
                n = node

                while n != -1:
                    path.append(n)
                    n = parents[n]

                if is_stable_path(path, t_m, t_n):
                    return path
                else:
                    path = []

        if not on_stack[child]:
            s.append((child, 0))
            parents[child] = node
            on_stack[child] = True

    return []



def potential_method(C, a, b, basis_solution, caps):
    m, n = C.shape
    print(f'>> Method of potentials')
    print(f'm={m}, n={n}')
    # Now we need to find ui, vj using:
    # ui + vj = cijB
    while True:
        basis_indices = list(map(tuple, np.argwhere(caps == 1)))
        non_basis_indices = list(map(tuple, np.argwhere(caps == 0)))
        print(f'basic indices: {basis_indices}')
        print(f'non basic indices: {non_basis_indices}')
        potentials_systemA = np.zeros((m + n - 1, m + n))
        potentials_systemB = np.zeros(m + n - 1)
        for row, base_coords in enumerate(basis_indices):
            base_i = base_coords[0]
            base_j = base_coords[1]
            potentials_systemA[row][base_i] = 1.0
            potentials_systemA[row][m + base_j] = 1.0
            potentials_systemB[row] = C[base_i][base_j]

        print(f'Initial potential system:')
        print(potentials_systemA)
        print(potentials_systemB)

        to_set_zero_index, to_set_zero_axis = find_best_potential(caps)
        print('We shall set', 'u' if to_set_zero_axis == 0 else 'v', f'_{to_set_zero_index}={0}')

        to_set_zero_actual_index = to_set_zero_index
        if to_set_zero_axis == 1:
            to_set_zero_index = m + to_set_zero_index

        potentials_systemA = np.delete(potentials_systemA, to_set_zero_actual_index, 1)

        print(f'System now:')
        print(potentials_systemA)
        print(potentials_systemB)

        potential_system_solution = np.linalg.solve(potentials_systemA, potentials_systemB)
        potential_system_solution = np.insert(potential_system_solution, to_set_zero_actual_index, 0)
        print(f'Potentials:')
        print(potential_system_solution)

        r = None
        s = None
        lowest_val = None
        for i, j in non_basis_indices:
            # Cij - ui - vj >= 0
            ui = potential_system_solution[i]
            vj = potential_system_solution[m + j]
            val = C[i][j] - ui - vj
            print(f'i={i}, j={j}, Cij={C[i][j]}, ui={ui}, vj={vj}')
            print(f'\tval={val}')
            if val < 0:
                if lowest_val is None or val < lowest_val:
                    lowest_val = val
                    r = i
                    s = j
        
        if lowest_val is None:
            print(f'Stop reached!')
            return basis_solution

        print(f'Choosing negative value C_{r}_{s} = {lowest_val}')

        graph_matrix = construct_graph(r, s, caps)

        print(f'Graph: ')
        print(graph_matrix)

        path = get_cycle(graph_matrix, r*n + s, m, n)
        print(path)

        cycle_indices = list(map(lambda x: np.unravel_index(x, (m, n)), path))
        print(f'Cycle indices: {cycle_indices}')

        min_i = None
        min_j = None
        min_val = None
        for base_i, base_j in basis_indices:
            if min_val is None or basis_solution[base_i][base_j] < min_val:
                min_val = basis_solution[base_i][base_j]
                min_i = base_i
                min_j = base_j

        print(f'{min_i, min_j} leaves basis.')
        

        
    

        
        



    

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



        
    
    
    

