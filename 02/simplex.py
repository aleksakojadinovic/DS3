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


def get_simplex_matrix(A, b, c, Q, P):
    m, n = A.shape
    s_matrix = np.copy(A)
    s_matrix = np.hstack((s_matrix, b.reshape(-1, 1)))
    s_matrix = np.vstack((s_matrix, np.hstack((c, 0.0))))
    return s_matrix

def simplex(A, b, c, Q, P, x0):
    if len(A) == 1:
        A = np.array([A])
    else:
        A = np.array(A)
    b = np.array(b)
    c = np.array(c)
    P = np.array(P)
    Q = np.array(Q)
    x0 = np.array(x0)

    simplex_matrix = np.hstack((A, b.reshape(-1, 1)))
    simplex_matrix = np.vstack((simplex_matrix, np.hstack((c, 0))))
    print(simplex_matrix_to_string(simplex_matrix))

    sm, sn = simplex_matrix.shape
    print('--STEP 1--')
    ### STEP 1
    # Transform target function f = \sum{c_px_p} + \sum{c_qx_q}
    # find u = (u1, ..., um) such that
    # ukp = c_p, p in P
    # each column kp is A[:, p].reshape(-1, 1)

    u_sysA = np.array(simplex_matrix[:-1, P[0]]).reshape(-1, 1)
    u_sysB = np.array(c[P[0]])
    for p in P[1:]:
        new_col = simplex_matrix[:-1, p].reshape(-1, 1)
        u_sysA = np.hstack((u_sysA, new_col))
        u_sysB = np.hstack((u_sysB, c[p]))

    u = np.linalg.solve(u_sysA, u_sysB)

    print(f'u = {u}')

    # pure_f = np.array([c[q] - np.dot(u, A[:, q]) for q in Q] + [np.dot(u, b)])
    pure_f = np.zeros(sn)
    for i in range(sn):
        if i in Q:
            pure_f[i] = c[i] - np.dot(u, simplex_matrix[:-1, i])
        elif i == sn - 1:
            pure_f[i] = np.dot(u, b)
        else:
            pure_f[i] = 0

    print(f'pure_f = {pure_f}')

    print('--STEP 2--')
    if (pure_f[Q] >= 0).all():
        print(f'Step 2 stop condition reached, solution is x0 = {x0}')
        return x0
    print(f'Step 2 stop condition not reached, proceed to step3')

    print('--STEP 3--')

    j = np.where(pure_f[Q] < 0)[0][0]
    print(f'Choosing j = {j}')

    y_sysA = u_sysA
    y_sysB = simplex_matrix[:-1, j]

    y = np.linalg.solve(y_sysA, y_sysB)
    print(f'y = {y}')

    y_extended = np.zeros(sn)
    at_p = 0
    for i in range(sn):
        if i in P:
            y_extended[i] = y[at_p]
            at_p += 1
        else:
            y_extended[i] = 0
    print(f'\t\textended y={y_extended}')

    print('--STEP 4 --')
    T_interval = portion.closed(-portion.inf, portion.inf)
    for i in P:
        # we're looking for an expression of the form
        # x0[i] - ty[i] >= 0
        # transforms to:
        # ty[i] <= x0[i]
        # Now it all depends on the sign of y[i]
        if y_extended[i] == 0:
            continue

        left_side = y_extended[i]
        right_side = x0[i]
        if (left_side > 0 and right_side > 0) or (left_side < 0 and right_side < 0):
            # Same signs, meaning t <= x0[i] / y[i]
            t_current = portion.closed(-portion.inf, right_side/left_side)
        else:
            # different signs, meaning t >= x0[i] / y[i]
            t_current = portion.closed(right_side/left_side, portion.inf)
        T_interval = T_interval & t_current


    if T_interval.upper == portion.inf or T_interval.upper < 0:
        print(f'Step 4 condition - f unbounded!')
        # maybe return None instead of this
        return float('-inf')


    t_star = T_interval.upper
    print(f't_star = {t_star}')


    print('--STEP 5--')
















def test1():
    c = [-1, -2, 0, 0]
    A = [[1, 1, 1, 0],
         [-1, 3, 0, 1]]
    b = [3, 5]
    Q = [0, 1]
    P = [2, 3]
    x0 = [0, 0, 3, 5]
    simplex(A, b, c, Q, P, x0)


test1()
