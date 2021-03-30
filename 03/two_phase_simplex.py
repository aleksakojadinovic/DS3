import numpy as np
import lp_input_parser as lpp
import dual_simplex as ds

tfLOG = True
DEC = 8

def tflog(*args, **kwargs):
    if tfLOG:
        print(*args, **kwargs)

"""
Creates a modified problem for phase one of modified simplex algorithm.
If the original simplex matrix dimensions are `m` and `n`
the modified dimensions are m x (n+m)
because the number of constraints does not change
and we add m artificial variables
"""
def get_modified_problem(simplex_matrix):
    m, n = simplex_matrix.shape
    A = simplex_matrix[:-1, :-1]
    b = simplex_matrix[:, -1]
    modified_simplex_matrix = np.zeros((m, n + m - 1), dtype=simplex_matrix.dtype)
    modified_simplex_matrix[:A.shape[0], :A.shape[1]] = A
    modified_simplex_matrix[:, -1] = b
    modified_simplex_matrix[:-1, A.shape[1]:-1] = np.eye(m - 1)
    modified_simplex_matrix[-1, n:] = np.ones(m-1)    
    return modified_simplex_matrix

def two_phase_simplex(simplex_matrix):

    ### PHASE 1
    tflog(f'Original:\r\n {simplex_matrix}')
    modified_simplex_matrix = get_modified_problem(simplex_matrix)    
    tflog(f'Modified:\r\n {modified_simplex_matrix}')

    omega, omega_val, omega_last_matrix = ds.dual_simplex(modified_simplex_matrix, True)
    tflog(f'omega solution: {omega_val}')
    tflog(f'omega last matrix:')
    tflog(omega_last_matrix)

    return None, None

arg_parser = lpp.get_simplex_parser()
args = vars(arg_parser.parse_args())
tfLOG = args['logging']
smatrix, _ = lpp.prepare_for_algorithm(args)

x, v = two_phase_simplex(smatrix)
# lpp.print_solution(args, x, v)