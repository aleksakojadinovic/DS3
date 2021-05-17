import numpy as np
import lp_input_parser as lpp
import argparse

duLOG = False
DEC = 8

def dslog(*args, **kwargs):
    if duLOG:
        print(*args, **kwargs)

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
    
    dslog('<<<<<<<<<<<<<<<<<<< STARTING DUAL SIMPLEX ALGORITHM >>>>>>>>>>>>>>>>>>>')
    k = 0
    while True:
        A = simplex_matrix[:-1, :-1]
        b = simplex_matrix[:-1, -1]
        c = simplex_matrix[-1, :-1]
        dslog(f'*****Iteration {k}')
        dslog('---K1')
        
        if (b >= 0).all():
            return None, np.round(-simplex_matrix[-1, -1], DEC)
        
        neg_indices = np.argwhere(b < 0).T[0]
        dslog(f'Neg indices: {neg_indices}')

        dslog('---K2')
        if np.apply_along_axis(np.all, 1, A[neg_indices]).any():
            dslog(f'\t Found positive row, no solution.')
            return None, None
                
        dslog('No positive column found, continue to K3.')

        dslog('---K3')
        s = neg_indices[-1]
        dslog(f'Choosing s = {s}')
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
            dslog(f"\t Couldn't find r_max, no solution.")
            return None, None


        dslog(f'r = {r} with Asr = {simplex_matrix[s][r]}')
        dslog('---K4')
     
        # Here we include the last one
        for i in range(canon_m):
            if i == s or simplex_matrix[i][r] == 0:
                continue
            coeff = -simplex_matrix[i][r] / simplex_matrix[s][r]
            new_row = simplex_matrix[s] * coeff + simplex_matrix[i]
            simplex_matrix[i] = new_row

        simplex_matrix[s] = simplex_matrix[s] / simplex_matrix[s][r]
        dslog('Updated simplex matrix to: ')
        dslog(simplex_matrix)
        k = k + 1

if __name__ == '__main__':
    arg_parser = lpp.get_simplex_parser()
    args = vars(arg_parser.parse_args())
    duLOG = args['logging']
    smatrix, _ = lpp.prepare_for_algorithm(args)

    x, v = dual_simplex(smatrix)
    lpp.print_solution(args, x, v)

