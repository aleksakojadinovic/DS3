import numpy as np
from scipy.optimize import linprog as lp


FLOAT_T = 'float64'

# Problem:

# Minimize cTx
# subject to constraints
# Ax = b
# where x >= 0, x in Z^n
# A in Z^m x Z^n
# b in Z^m
# c in Z^n

def fi(x):
    return x - np.floor(x)

def to_canonical(A, b, c):
    A = np.array(A)
    b = np.array(b)
    c = np.array(c)
    m, n = A.shape

    s_matrix = np.append(A, np.eye(m), axis=1)
    s_matrix = np.vstack((s_matrix, np.append(c, np.zeros(s_matrix.shape[1] - n))))
    s_matrix = np.hstack((s_matrix, np.append(b, 0).reshape(-1, 1)))

    return s_matrix



# NOTE: Assumes matrix in
def gomory_cut_method(c, eqA, eqb, leqA, leqb):
    
    # last_simplex_matrix = None
    def simplex_callback(x, **kwargs):
        print(kwargs['tabluea'])


    sol = lp(c, A_ub=leqA,
                b_ub=leqb,
                A_eq=eqA,
                b_eq=eqb,
                method='simplex',
                callback=simplex_callback)


    
        
def example1():
    c = [-1, -1, 0, 0]
    eqA = [[2, -2, -3, 2],
           [0, 3, 3, -1]]
    eqb = [5, 3]
    leqA = None
    leqb = None

    gomory_cut_method(c, eqA, eqb, leqA, leqb)


if __name__ == '__main__':
    example1()