import numpy as np

def pivot_basis(simplex_matrix, basic_indices, i0, j0):
    # We pivot around i0, j0, meaning we will get a new column that looks something like this 
    basic_indices = np.array(basic_indices)
    new_column = np.zeros(simplex_matrix.shape[0])
    new_column[i0] = 1.0

    found = False

    for ii, basic_index in enumerate(basic_indices):
        curr_basic_column = simplex_matrix[:, basic_index].copy()
        if (curr_basic_column == new_column).all():
            found = True
            basic_indices[ii] = j0
            break

    return list(basic_indices)

def reg_simplex(simplex_matrix, basic_indices):
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
            return True, simplex_matrix, basic_indices

        j0 = np.argwhere(c < 0)[0][0]
        


        if (simplex_matrix[:-1, j0] < 0).all():
            return False, None, None

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
            return False, None, None

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
        # if iteration == 3:
        #     return simplex_matrix, False

            

