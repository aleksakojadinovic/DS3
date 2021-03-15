import numpy as np
import portion
import sys
import argparse

LOG = True

def log(*args, **kwargs):
    if LOG:
        print(*args, **kwargs)


"""Linear expression to string"""
def lexp_to_string(exp):
    return ' + '.join(f'{c}*x_{i}' for i, c in enumerate(exp))

def lineq_to_string(a, b, sign):
    return lexp_to_string(a) + ' ' + sign + ' ' + str(b)

"""This will look very bad when numbers are not nice"""
def simplex_matrix_to_string(A):
    sep = '\t\t'
    sep2 = '\t\t\t'
    row_markers = sep2.join(f'x_{i}' for i, _ in enumerate(A[0][:-1]))
    row_markers += sep2 + '|b'
    row_split = '-'*len(row_markers)*2
    rows = []
    for row in A:
        regular_values = sep.join("{:.2f}".format(val) for val in row[:-1])
        rows.append(regular_values + sep + '|' + "{:.2f}".format(row[-1]))

    return row_markers + '\r\n' + row_split + '\r\n' + '\r\n'.join(rows[:-1]) + '\r\n' + row_split + '\r\n' + rows[-1]



"""Returns a tuple (simplex_matrix, Q, P, x0)"""
def to_canonical(A, b, c):
    # TODO: Optimize by allocating entire matrix once and then filling values with indexing
    # TODO: Also this is probably incorrect since basis doesn't always have to be consisted
    #       of the slack variables (I guess?) but I don't really get it :(
    A = np.array(A)
    b = np.array(b)
    c = np.array(c)
    m, n = A.shape

    s_matrix = np.append(A, np.eye(m), axis=1)
    s_matrix = np.vstack((s_matrix, np.append(c, np.zeros(s_matrix.shape[1] - n))))
    s_matrix = np.hstack((s_matrix, np.append(b, 0).reshape(-1, 1)))
    P = np.array(range(n, s_matrix.shape[1] - 1))
    Q = np.array(range(n))
    # TODO: Hardcoded for now until I figure it out

    return s_matrix, Q, P, np.append(np.zeros(s_matrix.shape[1] - len(b) - 1), b)

def canonical_simplex(simplex_matrix, Q, P, x0):
    log('>>>>>>>>>>>>>>>>REVISED SIMPLEX ALGORITHM<<<<<<<<<<<<<<<<<<<<<<')
    iteration = 1
    while True:
        P = np.sort(P)
        Q = np.sort(Q)
        log(f'<<<Iteration {iteration}>>>')
        log(simplex_matrix_to_string(simplex_matrix))
        iteration += 1
        log(f'P={P}, Q={Q}, x0={x0}')

        b = simplex_matrix[:-1, -1]
        c = simplex_matrix[-1:, :].flatten()
        simplex_m, simplex_n = simplex_matrix.shape
        num_basis = len(P)
        num_non_basis = len(Q)

        log('------STEP 1-------')
        u_sysA = np.array(simplex_matrix[:-1, P]).T
        u_sysB = np.array(c[P])
        u = np.linalg.solve(u_sysA, u_sysB)
        log(f'u = {u}')

        pure_f = np.zeros(simplex_n)
        pure_f[-1] = np.dot(u, b)
        for i in range(len(pure_f - 1)):
            pure_f[i] = c[i] - np.dot(u, simplex_matrix[:-1, i]) if i in Q else 0
        log(f'pure f = {pure_f}')

        log('------STEP 2-------')
        if (pure_f[:-1] >= 0).all():
            log(f'Step 2 stop condition reached, returning current x0')
            return x0, np.dot(x0, c[:-1])
        log('Step 2 stop condition NOT reached, continue to step 3')

        log('------STEP 3-------')
        j = np.where(pure_f[:-1] < 0)[0][0]

        log(f'Choosing j = {j}')

        y_sysA = np.array(simplex_matrix[:-1, P])
        y_sysB = np.array(simplex_matrix[:-1, j])

        y_sol = np.linalg.solve(y_sysA, y_sysB)
        log(f'y = {y_sol}')
        y = np.zeros(simplex_n)
        y[P] = y_sol
        log(f'y extended = {y}')

        log('------STEP 4-------')
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
            log(f'Step 4 stop condition reached, function is unbounded')
            return None
        t = t_interval.upper
        log(f'Choosing t = {t}')

        log('------STEP 5-------')
        s = None
        for s_candidate in P:
            if y[s_candidate] <= 0:
                continue
            if x0[s_candidate] - t*y[s_candidate] == 0:
                s = s_candidate
                break

        if s is None:
            log(f'Step 5 - s not found, no solution (I guess?)')
            return None

        log(f'Choosing s = {s}')

        x1 = np.zeros(len(x0))
        x1[j] = t
        for i in P[P != s]:
            x1[i] = x0[i] - t * y[i]
        newP = np.append(P[P != s], j)
        newQ = np.append(Q[Q != j], s)
        log(f'x1 = {x1}')
        log(f'new P = {newP}')
        log(f'new Q = {newQ}')

        x0 = x1
        P = newP
        Q = newQ


def test1():
    A = [[1, 1],
        [-1, 3]]
    b = [3, 5]
    c = [-1, -2]
    s_matrix, Q, P, x0 = to_canonical(A, b, c)
    sol = canonical_simplex(s_matrix, Q, P, x0)
    print(sol)


test1()

