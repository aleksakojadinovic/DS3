def reg_simplex(simplex_matrix):
    sim_m, sim_n = simplex_matrix.shape
    for i in range(sim_m - 1):
        if simplex_matrix[i, -1] < 0:
            simplex_matrix[i] *= -1

    print(f'Now starting regular simplex for: ')
    print(simplex_matrix)


    while True:
        c = simplex_matrix[-1, :-1]
        b = simplex_matrix[:-1, -1]
        if (c >= 0).all():
            return simplex_matrix, True

        j0 = np.argwhere(c < 0)[0]
        
        if (simplex_matrix[:-1, j0] < 0).all():
            return simplex_matrix, False

        i0 = None
        piv_min = None
        for i in range(sim_m - 1):
            if simplex_matrix[i, j0] <= 0:
                continue

            piv_curr = b[i] / simplex_matrix[i][j0]
            if piv_min is None or piv_curr < piv_min:
                i0 = i
                piv_min = piv_curr

        

        ai0j0 = simplex_matrix[i0][j0]

        for i in range(i0):
            #make this one zero
            pass

            

