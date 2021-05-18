import sys

sys.path.append('../_simplex_utils')
from lp_utils import swap_basis
from lp_utils import fetch_sol_from_simplex_matrix

import numpy as np
import pandas as pd


def tableau_simplex(simplex_matrix, basic_indices, phase=None):

    simplex_matrix = np.array(simplex_matrix)
    basic_indices = np.array(basic_indices)

    for i in range(len(simplex_matrix) - 1):
        if simplex_matrix[i][-1] < 0:
            simplex_matrix[i] *= -1


    sim_m, sim_n = simplex_matrix.shape
    # for i in range(sim_m - 1):
    #     if simplex_matrix[i, -1] < 0:
    #         simplex_matrix[i] *= -1

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
        basic_indices = swap_basis(simplex_matrix, basic_indices, i0, j0)

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


            

