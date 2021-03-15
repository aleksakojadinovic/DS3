import numpy as np
import portion
# Linear programming problem:
# Minimize f = c1x1 + ... + cnxn
# given:
# k1x1 + ... knxn = b (k are columns)
# and:
# x1, .... xn >= 0

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
    return s_matrix, Q, P, np.array([0, 0, 3, 5])

def canonical_simplex(simplex_matrix, Q, P, x0):
    print('>>>>>>>>>>>>>>>>CANONICAL SIMPLEX ALGORITHM<<<<<<<<<<<<<<<<<<<<<<')
    iteration = 1
    while True:
        P = np.sort(P)
        Q = np.sort(Q)
        print(f'<<<Iteration {iteration}>>>')
        print(simplex_matrix_to_string(simplex_matrix))
        iteration += 1
        print(f'P={P}, Q={Q}, x0={x0}')

        b = simplex_matrix[:-1, -1]
        c = simplex_matrix[-1:, :].flatten()
        simplex_m, simplex_n = simplex_matrix.shape
        num_basis = len(P)
        num_non_basis = len(Q)

        print('------STEP 1-------')
        u_sysA = np.array(simplex_matrix[:-1, P]).T
        u_sysB = np.array(c[P])
        print(f'Solving system for u with', u_sysA, u_sysB, sep='\r\n')
        u = np.linalg.solve(u_sysA, u_sysB)
        print(f'u = {u}')

        pure_f = np.zeros(simplex_n)
        pure_f[-1] = np.dot(u, b)
        for i in range(len(pure_f - 1)):
            pure_f[i] = c[i] - np.dot(u, simplex_matrix[:-1, i]) if i in Q else 0
        print(f'pure f = {pure_f}')

        print('------STEP 2-------')
        if (pure_f[:-1] >= 0).all():
            print(f'Step 2 stop condition reached, returning current x0')
            return x0, np.dot(x0, c[:-1])
        print('Step 2 stop condition NOT reached, continue to step 3')

        print('------STEP 3-------')
        j = np.where(pure_f[:-1] < 0)[0][0]
        print(f'Choosing j = {j}')
        print('Now solve the system for y: ')
        y_sysA = np.array(simplex_matrix[:-1, P])
        y_sysB = np.array(simplex_matrix[:-1, j])
        print(y_sysA, y_sysB)
        y_sol = np.linalg.solve(y_sysA, y_sysB)
        print(f'y = {y_sol}')
        y = np.zeros(simplex_n)
        y[P] = y_sol
        print(f'y extended = {y}')

        print('------STEP 4-------')
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
            print(f'Step 4 stop condition reached, function is unbounded')
            return None
        t = t_interval.upper
        print(f'Choosing t = {t}')

        print('------STEP 5-------')
        s = None
        for s_candidate in P:
            if y[s_candidate] <= 0:
                continue

            # xs =
            if x0[s_candidate] - t*y[s_candidate] == 0:
                s = s_candidate
                break

        if s is None:
            print(f'Step 5 - s not found, no solution (I guess?)')
            return None

        print(f'Choosing s = {s}')

        x1 = np.zeros(len(x0))
        x1[j] = t
        for i in P[P != s]:
            x1[i] = x0[i] - t * y[i]
        newP = np.append(P[P != s], j)
        newQ = np.append(Q[Q != j], s)
        print(f'x1 = {x1}')
        print(f'new P = {newP}')
        print(f'new Q = {newQ}')

        x0 = x1
        P = newP
        Q = newQ

        simplex_matrix[:-1, s] = y_sol





def test1():
    c = [-1, -2]
    A = [[1, 1,],
         [-1, 3]]
    b = [3, 5]

    s_matrix, Q, P, x0 = to_canonical(A, b, c)
    sol = canonical_simplex(s_matrix, Q, P, x0)
    print(sol)


def test2():
    A = [[1, -1, -1, 3, 1, 0, 0, 1],
         [5, 1, 3, 8, 0, 1, 0, 55],
         [-1, 2, 3, -5, 0, 0, 1, 4],
         [-4, -1, -5, -3, 0, 0, 0, 0]]
    x0 = [0, 0, 0, 0, 1, 55, 3]
    P = [4, 5, 6]
    Q = [0, 1, 2, 3]
    sol = canonical_simplex(np.array(A), np.array(Q), np.array(P), np.array(x0))
    print(sol)

test1()
