import numpy as np
import portion as P

def mvstack(tup):
    a, b = tup
    if len(a) == 0:
        return np.array(b)
    else:
        return np.vstack((a, b))


def remove_variable(A, b, idx):
    # Removes variable at index `idx`
    # from system Ax >= b
    m, n = A.shape

    # List of stuff that was removed,
    # it's a list of pairs (np.array, bool)
    # where the first element is the expression
    # on the right side and the second element
    # is the indicator whether it's >= or <= (>= is True)
    removed = []

    # List of inequalities that did not contain this variable in the first place
    unmodifiedA = np.array([])
    unmodifiedB = np.array([])

    for sum_part, constant in zip(A, b):
        if np.isclose(sum_part[idx], 0.0):
            unmodifiedA = mvstack((unmodifiedA, sum_part))
            unmodifiedB = np.hstack((unmodifiedB, constant))
        else:
            coeff = sum_part[idx]
            right_part = np.hstack((-sum_part/coeff, constant/coeff))
            right_part[idx] = 0.0
            greater    = sum_part[idx] > 0

            removed.append((right_part, greater))

    new_ineqsA = np.array([])
    new_ineqsb = np.array([])
    for i in range(len(removed)):
        expr1 = removed[i][0]
        sign1 = removed[i][1]
        for j in range(i+1, len(removed)):
            expr2 = removed[j][0]
            sign2 = removed[j][1]

            if sign1 == sign2:
                continue

            # The one that x is greater than is the one that goes on the left
            left_one  = expr1 if sign1 else expr2
            # The one that x is less than is the one that goes on the right
            right_one = expr1 if sign2 else expr2

            # Now we know that
            # left_one <= x <= right_one
            # meaning
            # left_one <= right_one
            # eqv to
            # right_one - left_one >= 0

            new_ineq = right_one - left_one
            new_ineqA  = new_ineq[:-1]
            new_ineqb = new_ineq[-1]

            new_ineqsA = mvstack((new_ineqsA, new_ineqA))
            new_ineqsb = np.hstack((new_ineqsb, -new_ineqb))

    newA = mvstack((unmodifiedA, new_ineqsA))
    newb = np.hstack((unmodifiedB, new_ineqsb))
    return newA, newb

def fourier_motzkin(A, b, varnames=None):

    A = np.array(A, dtype='float32')
    b = np.array(b, dtype='float32')
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError(f'Invalid `b` size, expecting {m}')

    if varnames is not None and len(varnames) != n:
        raise ValueError(f'Expecting {n} varnames, got {len(varnames)}')
    if varnames is None:
        varnames = [f'x_{i}' for i in range(n)]


    variables = set()
    for expr in A:
        # TODO: Refactor (np.unique, np.argwhere and something like that)
        for idx in map(lambda x: x[0], filter(lambda p: not np.isclose(p[1], 0), enumerate(expr))):
            variables.add(idx)


    currA = np.array(A)
    currB = np.array(b)
    for var_to_remove in list(variables)[:-1]:
        currA, currB = remove_variable(currA, currB, var_to_remove)
    final_var = list(variables)[-1]

    # For testing:
    # for var_to_remove in [1, 2]:
    #     currA, currB = remove_variable(currA, currB, var_to_remove)
    # final_var = 0


    final_interval = P.closed(-P.inf, P.inf)

    for a, b in zip(currA, currB):
        coeff = a[final_var]
        val = b/coeff
        if coeff > 0:
            interval = P.closed(val, P.inf)
        else:
            interval = P.closed(-P.inf, val)
        final_interval = final_interval & interval

    print(f'{final_interval.lower} <= {varnames[final_var]} <= {final_interval.upper}')




mA = [[7, 2, -2],
     [-1, -1, -1],
     [-2, 3, 1],
     [5, -1, 1]]

mb = [4, -4, 1, -2]

fourier_motzkin(mA, mb, varnames=['x', 'y', 'z'])
