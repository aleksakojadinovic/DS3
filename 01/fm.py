import numpy as np
import portion as P

# Util, same as np.vstack but it stack onto an empty array
def mvstack(tup):
    a, b = tup
    if len(a) == 0:
        return np.array(b)
    else:
        return np.vstack((a, b))

# Removes variable at index `idx`
# from system Ax >= b
def remove_variable(A, b, idx):
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
    new_ineqsB = np.array([])
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
            new_ineqsB = np.hstack((new_ineqsB, -new_ineqb))

    newA = mvstack((unmodifiedA, new_ineqsA))
    newB = np.hstack((unmodifiedB, new_ineqsB))
    return newA, newB

# Returns a closed interval for x_idx
# in system Ax >= b that only contains x_idx
def solve_single(A, b, idx):
    final_interval = P.closed(-P.inf, P.inf)

    for a, b in zip(A, b):
        coeff = a[idx]
        val = b / coeff
        if coeff > 0:
            interval = P.closed(val, P.inf)
        else:
            interval = P.closed(-P.inf, val)
        final_interval = final_interval & interval
    return final_interval

# Substitues x_id with value in Ax >= b
def substitute(A, b, idx, value):
    # Inequalities are of form
    # c0x0 + c1x1 + ... c_idx*x_idx + ... + c_n-1x_n-1 >= b
    # Substitute value:
    # c0x0 + c1x1 + ... c_idx*value + ... + c_n-1x_n-1 >= b
    # Move to right side:
    # c0x0 + c1x1 + ... 0x_idx + ... + c_n-1x_n-1 >= b - c_idx*value
    nA = np.array(A)
    nb = np.array(b)

    for i in range(len(A)):
        nb[i] = nb[i] - nA[i][idx] * value
        nA[i][idx] = 0

    return nA, nb


def fourier_motzkin(A, b, varnames=None):

    A = np.array(A, dtype='float32')
    b = np.array(b, dtype='float32')
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError(f'Invalid `b` size, expecting {m}')

    if varnames is not None:
        if len(varnames) != n:
            raise ValueError(f'Expecting {n} varnames, got {len(varnames)}.')
        if len(set(varnames)) != len(varnames):
            raise ValueError(f'Varnames must to be unqiue.')
    else:
        varnames = [f'x_{i}' for i in range(n)]

    variables = set()
    for expr in A:
        # TODO: Refactor (np.unique, np.argwhere and something like that)
        for idx in map(lambda x: x[0], filter(lambda p: not np.isclose(p[1], 0), enumerate(expr))):
            variables.add(idx)
    variables = list(variables)

    # List of system through removals

    currA = np.array(A)
    currB = np.array(b)
    history = [(np.array(currA), np.array(currB))]

    for var_to_remove in variables[:-1]:
        currA, currB = remove_variable(currA, currB, var_to_remove)
        history.append((np.array(currA), np.array(currB)))

    final_var = list(variables)[-1]

    final_interval = solve_single(currA, currB, final_var)
    if final_interval.empty:
        print(f'No solution found.')
        return None

    print(f'Interval for final variable: ')
    print(f'{final_interval.lower} <= {varnames[final_var]} <= {final_interval.upper}')

    chosen_var_values = np.array([None for _ in variables])
    chosen_var_values[-1] = any_in_interval(final_interval)

    chosen_intervals = np.array([None for _ in variables])
    chosen_intervals[-1] = final_interval


    for ii in range(n-2, -1, -1):
        i = variables[ii]
        print(f'Finding range for variable x_{i} or rather {varnames[i]}')
        print(f'This can be achieved by substituting the following variables: ')
        # print(', '.join([f'{varnames[j]}' for j in range()]))
        # from m-1 backwards and take i+1 of them
        vars_to_sub = [j for j in range(i+1, n)]
        values_to_sub = [chosen_var_values[k] for k in vars_to_sub]
        print(", ".join(f'x_{thei} = {thev}' for thei, thev in zip(vars_to_sub, values_to_sub)))

        tmpA = history[i][0]
        tmpB = history[i][1]

        print('Into system: ')
        print(tmpA, tmpB)

        tmpA = history[i][0]
        tmpB = history[i][1]
        for var_idx, val in zip(vars_to_sub, values_to_sub):
            print(f'\t\tsub...x_{var_idx} or rather {varnames[var_idx]} with value {val}')
            tmpA, tmpB = substitute(tmpA, tmpB, var_idx, val)

        print(f'After substitution we have: ')
        print(tmpA, tmpB)

        target_solution = solve_single(tmpA, tmpB, i)
        if target_solution.empty:
            print(f'No solution found.')
            return None

        chosen_var_values[i] = any_in_interval(target_solution)
        chosen_intervals[i] = target_solution



        print('===================================')

    print('-----------------------------------------------------------------------------')
    for i, sol_interval in enumerate(chosen_intervals):
        print(f'{varnames[i]} is in {sol_interval}')









def any_in_interval(interval):
    if interval.empty:
        raise ValueError(f'Empty interval')
    if interval.lower == -P.inf and interval.upper == P.inf:
        return 0.0
    if interval.lower == -P.inf:
        return interval.upper
    if interval.upper == P.inf:
        return interval.lower

    return interval.upper


mA = [[7, 2, -2],
     [-1, -1, -1],
     [-2, 3, 1],
     [5, -1, 1]]

mb = [4, -4, 1, -2]

fourier_motzkin(mA, mb, varnames=['x', 'y', 'z'])
