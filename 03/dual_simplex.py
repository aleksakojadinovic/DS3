import numpy as np
import lp_input_parser as lpp
import argparse

LOG = True
DEC = 8

def log(*args, **kwargs):
    if LOG:
        print(*args, **kwargs)

"""
f - Target function c0, ... c_{n-1}
A - m x n constraint matrix
b - right side vector
"""
def dual_simplex(simplex_matrix, gimme_matrix=False):
    simplex_matrix = np.array(simplex_matrix, dtype='float64')
    canon_m, canon_n = simplex_matrix.shape
    m = canon_m - 1
    n = canon_n - 1
    
    log('<<<<<<<<<<<<<<<<<<< STARTING DUAL SIMPLEX ALGORITHM >>>>>>>>>>>>>>>>>>>')
    k = 0
    while True:
        A = simplex_matrix[:-1, :-1]
        b = simplex_matrix[:-1, -1]
        c = simplex_matrix[-1, :-1]
        log(f'*****Iteration {k}')
        log('---K1')
        
        if (b >= 0).all():
            log(f'\t All bs are positive, optimal found.')
            if gimme_matrix:
                return None, np.round(-simplex_matrix[-1, -1], DEC), simplex_matrix
            else:
                return None, np.round(-simplex_matrix[-1, -1], DEC)
        
        neg_indices = np.argwhere(b < 0).T[0]
        log(f'Neg indices: {neg_indices}')

        log('---K2')
        if np.apply_along_axis(np.all, 1, A[neg_indices]).any():
            log(f'\t Found positive row, no solution.')
            if gimme_matrix:
                return None, None, simplex_matrix
            else:
                return None, None
                
        log('No positive column found, continue to K3.')

        log('---K3')
        # TODO: Add Blend flag 
        s = neg_indices[-1]
        log(f'Choosing s = {s}')
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
            log(f"\t Couldn't find r_max, no solution.")
            if gimme_matrix:
                return None, None, simplex_matrix
            else:
                return None, None


        log(f'r = {r} with Asr = {simplex_matrix[s][r]}')

        log('---K4')
     
        # Here we include the last one
        for i in range(canon_m):
            if i == s or simplex_matrix[i][r] == 0:
                continue
            coeff = -simplex_matrix[i][r] / simplex_matrix[s][r]
            new_row = simplex_matrix[s] * coeff + simplex_matrix[i]
            simplex_matrix[i] = new_row

        simplex_matrix[s] = simplex_matrix[s] / simplex_matrix[s][r]
        log('Updated simplex matrix to: ')
        log(simplex_matrix)
        k = k + 1


arg_parser = lpp.get_simplex_parser()
args = vars(arg_parser.parse_args())
LOG = args['logging']
smatrix, _ = lpp.prepare_for_algorithm(args)

x, v = dual_simplex(smatrix)
lpp.print_solution(args, x, v)

