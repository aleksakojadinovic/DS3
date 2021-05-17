import numpy as np
import lp_input_parser as lpp
import argparse

duLOG = False
DEC = 8

"""
f - Target function c0, ... c_{n-1}
A - m x n constraint matrix
b - right side vector
"""
def dual_simplex(simplex_matrix):
    simplex_matrix = np.array(simplex_matrix, dtype='float64')
    canon_m, canon_n = simplex_matrix.shape
    m = canon_m - 1
    n = canon_n - 1
    
    k = 0
    while True:
        A = simplex_matrix[:-1, :-1]
        b = simplex_matrix[:-1, -1]
        c = simplex_matrix[-1, :-1]
        
        if (b >= 0).all():
            return None, np.round(-simplex_matrix[-1, -1], DEC)
        
        neg_indices = np.argwhere(b < 0).T[0]

        if np.apply_along_axis(np.all, 1, A[neg_indices]).any():
            return None, None

        s = neg_indices[-1]
        # Now find 
        # max over r of {cr / A_sr such that Asr > 0}
        r = None
        r_max_val = None
        for r_cand in range(n):
            if simplex_matrix[s][r_cand] >= 0:
                continue
            val = c[r_cand]/simplex_matrix[s][r_cand]
            if r is None or val > r_max_val:
                r = r_cand
                r_max_val = val

        if r is None:
            return None, None

     
        # Here we include the last one
        for i in range(canon_m):
            if i == s or simplex_matrix[i][r] == 0:
                continue
            coeff = -simplex_matrix[i][r] / simplex_matrix[s][r]
            new_row = simplex_matrix[s] * coeff + simplex_matrix[i]
            simplex_matrix[i] = new_row

        simplex_matrix[s] = simplex_matrix[s] / simplex_matrix[s][r]
        k = k + 1

if __name__ == '__main__':
    arg_parser = lpp.get_simplex_parser()
    args = vars(arg_parser.parse_args())
    duLOG = args['logging']
    smatrix, _ = lpp.prepare_for_algorithm(args)

    x, v = dual_simplex(smatrix)
    lpp.print_solution(args, x, v)

