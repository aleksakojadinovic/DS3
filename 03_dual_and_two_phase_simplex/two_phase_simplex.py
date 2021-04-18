import numpy as np
import lp_input_parser as lpp
import dual_simplex as ds
import simplex as revised_simplex
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
    c = simplex_matrix[-1, :]
    modified_simplex_matrix = np.zeros((m, n + m - 1), dtype=simplex_matrix.dtype)
    modified_simplex_matrix[:A.shape[0], :A.shape[1]] = A
    modified_simplex_matrix[:, -1] = b
    modified_simplex_matrix[:-1, A.shape[1]:-1] = np.eye(m - 1)
    print(modified_simplex_matrix)

    modified_simplex_matrix[-1, n-1:] = c[:-1]
    return modified_simplex_matrix


def two_phase_simplex(simplex_matrix):
    simplex_matrix = np.array(simplex_matrix, dtype='float64')
    m, n = simplex_matrix.shape
    num_vars = n - 1
    ### PHASE 1
    tflog('--PHASE 1')
    tflog(f'Original:\r\n {simplex_matrix}')
    modified_simplex_matrix = get_modified_problem(simplex_matrix)    
    tflog(f'Modified:\r\n {modified_simplex_matrix}')

    _, omega_val, omega_last_matrix = ds.dual_simplex(modified_simplex_matrix, True)
    print(omega_last_matrix)
    # if omega_val != 0:
    #     tflog(f'Omega val is {omega_val}, stopping.')
    #     return None, None

    # ### PHASE 2
    # tflog('--PHASE 2')
    # tflog('Last known omega matrix:')
    # tflog(omega_last_matrix)

    # # Now find indices of all basis columns
    # omega_m, omega_n = omega_last_matrix.shape
    # basis_indices = []
    # for col_index in range(omega_n - 1):
    #     candidate_column = omega_last_matrix[:-1, col_index]
    #     if len(candidate_column[candidate_column == 1]) == 1 and len(candidate_column[candidate_column == 0]) == len(candidate_column) - 1:
    #         tflog(f'\t{candidate_column} is basic.')
    #         basis_indices.append(col_index)

    # if len(basis_indices) != m - 1:
    #     raise ValueError(f'**************** COMES RIGHT BACK')

    # tflog(f'Columns with indices: {basis_indices} form a max-rank basis submatrix.')
            
        
    




    return None, None

if __name__ == '__main__':
    arg_parser = lpp.get_simplex_parser()
    args = vars(arg_parser.parse_args())
    tfLOG = args['logging']
    smatrix, _ = lpp.prepare_for_algorithm(args)

    x, v = two_phase_simplex(smatrix)
    lpp.print_solution(args, x, v)