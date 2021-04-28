import numpy as np
import portion
import sys

DEC = 8

def canonical_simplex(simplex_matrix, Q, P, x0):
    
    iteration = 1
    simplex_m, simplex_n = simplex_matrix.shape

    b = simplex_matrix[:-1, -1]
    c = simplex_matrix[-1:, :].flatten()

    while True:
        iteration += 1
        u = np.linalg.solve(simplex_matrix[:-1, P].T, c[P])
        pure_f = np.zeros(simplex_n)
        pure_f[-1] = np.dot(u, b)
        
        for i in range(len(pure_f) - 1):
            pure_f[i] = c[i] - np.dot(u, simplex_matrix[:-1, i]) if i in Q else 0
        if (pure_f[:-1] >= 0).all():
            return np.around(x0, DEC), np.round(np.dot(x0, c[:-1]), DEC), P, Q, iteration



        j = np.where(pure_f[:-1] < 0)[0][0]        

        y_sol = np.linalg.solve(np.array(simplex_matrix[:-1, P]), simplex_matrix[:-1, j])
        y = np.zeros(simplex_n)
        y[P] = y_sol

        t_interval = portion.closed(-portion.inf, portion.inf)
        for i in P:
            left_side = y[i]
            right_side = x0[i]
            if left_side == 0:
                continue
            if (left_side < 0 and right_side <= 0) or (left_side > 0 and right_side >= 0):
                c_interval = portion.closed(-portion.inf, right_side/left_side)
            else:
                c_interval = portion.closed(right_side/left_side, portion.inf)

            t_interval &= c_interval

        if t_interval.upper < 0 or t_interval.upper == portion.inf:
            return None, None, P, Q, iteration
        t = t_interval.upper

        s = None
        for s_candidate in P:
            if y[s_candidate] <= 0:
                continue
            if x0[s_candidate] - t*y[s_candidate] == 0:
                s = s_candidate
                break
        
        if s is None:
            return None, None, P, Q, iteration

        x1 = np.zeros(len(x0))
        x1[j] = t
        for i in P[P != s]:
            x1[i] = x0[i] - t * y[i]
        

        newQ = np.append(Q[Q != j], s)
        newP = np.where(P == s, j, P)

        x0 = x1
        P = newP
        Q = newQ


