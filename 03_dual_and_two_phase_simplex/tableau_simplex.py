import numpy as np

def pivot_basis(simplex_matrix, basic_indices, i0, j0):
    basic_indices = np.array(basic_indices)

    found = False    

    for ii, basic_index in enumerate(basic_indices):
        
        curr_basic_column = simplex_matrix[:-1, basic_index].copy()

        curr_basic_non_zero = np.invert(np.isclose(curr_basic_column, 0))
        curr_non_zero_idx_idces = np.argwhere(curr_basic_non_zero)[0]
        curr_non_zero_idx = curr_non_zero_idx_idces[0]
        
    
        
        if curr_non_zero_idx == i0:
            found = True
            basic_indices[ii] = j0
            break

    if not found:
        print('ovde puklo')
        raise ValueError(f'Means my algorithm is wrong because I cannot find leaving basis')

    return list(basic_indices)

def fetch_sol_from_simplex_matrix(simplex_matrix, basic_indices):   
    m, n = simplex_matrix.shape

    solution = np.zeros(n-1)
    for j in basic_indices:
        

        row_idx = np.argwhere(np.invert(np.isclose(simplex_matrix[:-1, j], 0)))[0][0]
        coeff = simplex_matrix[row_idx, j]
        solution[j] = simplex_matrix[row_idx, -1] / coeff

    return solution

def tableau_simplex(simplex_matrix, basic_indices, phase=None):

    simplex_matrix = np.array(simplex_matrix)
    basic_indices = np.array(basic_indices)

    for i in range(len(simplex_matrix) - 1):
        if simplex_matrix[i][-1] < 0:
            simplex_matrix[i] *= -1


    sim_m, sim_n = simplex_matrix.shape
    for i in range(sim_m - 1):
        if simplex_matrix[i, -1] < 0:
            simplex_matrix[i] *= -1

    iteration = 1
    

    while True:

        c = simplex_matrix[-1, :-1]
        b = simplex_matrix[:-1, -1]
        if (c >= 0).all():
            simplex_result = dict()
            simplex_result['bounded']       = True
            simplex_result['message']       = 'Successfully found optimum value'
            simplex_result['opt_val']       = -simplex_matrix[-1, -1]
            simplex_result['opt_point']     = fetch_sol_from_simplex_matrix(simplex_matrix, basic_indices)
            simplex_result['last_matrix']   = simplex_matrix
            simplex_result['basic_indices'] = basic_indices
            simplex_result['phase']     = phase
            return simplex_result

        j0 = np.argwhere(c < 0)[0][0]
        


        if (simplex_matrix[:-1, j0] < 0).all():
            simplex_result = dict()
            simplex_result['bounded'] = False
            simplex_result['message'] = 'The function is unbounded (no positive val in pivot column)'
            simplex_result['opt_val']     = 0.0
            simplex_result['opt_point']   = np.zeros(sim_n - 1)
            simplex_result['last_matrix'] = simplex_matrix
            simplex_result['basic_indices'] = basic_indices
            simplex_result['phase'] = phase
            return simplex_result

        i0 = None
        piv_min = None
        for i in range(sim_m - 1):
            if simplex_matrix[i, j0] <= 0:
                continue

            piv_curr = b[i] / simplex_matrix[i][j0]
            if piv_min is None or piv_curr < piv_min:
                i0 = i
                piv_min = piv_curr

        if i0 is None:
            simplex_result = dict()
            simplex_result['bounded'] = False
            simplex_result['message'] = 'The function is unbounded (no non positive val in pivot row)'
            simplex_result['opt_val']     = 0.0
            simplex_result['opt_point'] = np.zeros(sim_n - 1)
            simplex_result['last_matrix'] = simplex_matrix
            simplex_result['basic_indices'] = basic_indices
            simplex_result['phase'] = phase
            return simplex_result

        ai0j0 = simplex_matrix[i0][j0]
        basic_indices = pivot_basis(simplex_matrix, basic_indices, i0, j0)

        for i in range(sim_m):

            if i == i0:
                continue

            current_row = simplex_matrix[i]
            
            if current_row[j0] == 0:
                continue

            coeff = - current_row[j0] / ai0j0
            simplex_matrix[i] = coeff * simplex_matrix[i0] + current_row

        simplex_matrix[i0] /= ai0j0


        iteration += 1


            

