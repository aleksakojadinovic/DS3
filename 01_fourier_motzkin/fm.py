import numpy as np
import portion as P

# ======================== UTILS ======================================

FLOAT_T = 'float32'
# FLOAT_T = 'float64'

# Util, same as np.vstack but it can stack onto an empty array
def mvstack(tup):
    a, b = tup
    if len(a) == 0:
        return np.array(b)
    else:
        return np.vstack((a, b))

# Inequality to string
def intos(a, b):
    return ' + '.join([f'{c}*x_{i}' for i, c in enumerate(a)]) + ' >= ' + str(b)

# Inequality system to string
def systos(A, b):
    return '\r\n'.join([intos(a, b) for a, b in zip(A, b)])

def misclose(a, b):
    return a == b

# ======================== MIDSTEPS ===================================

# Removes variable at index `idx`
# from system Ax >= b
def remove_variable(A, b, idx):
    m, n = A.shape

    # print(f'======REMOVING VARIABLE x_{idx}========')
    # print('from system: ')
    # print(systos(A, b))


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
        if misclose(sum_part[idx], 0.0):
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

    newA = unmodifiedA
    newB = unmodifiedB
    if len(new_ineqsA) > 0:
        newA = mvstack((unmodifiedA, new_ineqsA))
        newB = np.hstack((unmodifiedB, new_ineqsB))

    if len(newB) == 1:
        newA = np.array([newA])

    # print('i wanna return')
    # print(newA)
    # print(newB)
    return newA, newB

# Returns a closed interval for x_idx
# in system Ax >= b that only contains x_idx
def solve_single(A, b, idx):

    final_interval = P.closed(-P.inf, P.inf)
    for a, b in zip(A, b):
        # print(f'indexing with {idx} on {a}')
        coeff = a[idx]
        if misclose(coeff, 0.0):
            continue
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

# Picks a value from the interval
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

# ============================ (1) FOURIER-MOTZKIN ===============================

def fourier_motzkin(A, b, elimination_order=None, value_picks=None, just_last=False):

    A = np.array(A, dtype=FLOAT_T)
    b = np.array(b, dtype=FLOAT_T)
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError(f'Invalid `b` size, expecting {m}')

    if elimination_order is None:
        elimination_order = list(range(n))
    else:
        if any([i > n - 1 or i < 0 for i in elimination_order]):
            raise ValueError(f'Variable indices out of bounds in elimination order.')
        if len(set(elimination_order)) != len(elimination_order):
            raise ValueError(f'Duplicates in elimination order.')

    if value_picks is None:
        value_picks = 'any'

    varnames = [f'x_{i}' for i in range(n)]


    # List of system through removals
    currA = np.array(A)
    currB = np.array(b)


    history = [(np.array(currA), np.array(currB))]

    for var_to_remove in elimination_order[:-1]:
        currA, currB = remove_variable(currA, currB, var_to_remove)
        history.append((np.array(currA), np.array(currB)))



    final_var = list(elimination_order)[-1]

    final_interval = solve_single(currA, currB, final_var)
    if final_interval.empty:
        return None



    chosen_var_values = np.array([None for _ in elimination_order])
    chosen_var_values[final_var] = any_in_interval(final_interval) if value_picks == 'any' else value_picks[0]

    chosen_intervals = np.array([None for _ in elimination_order])
    chosen_intervals[final_var] = final_interval

    if just_last:
        return final_interval

    # print('----------------------------------------------------')
    # print(f'Interval for final variable: ')
    # print(f'{final_interval.lower} <= {varnames[final_var]} <= {final_interval.upper}')
    # print(f'Chosen value for final variable: ')
    # print(f'x_{final_var} = {chosen_var_values[final_var]}')
    # print('----------------------------------------------------')

    # Loops over variables that are to be found.
    for pick, i in enumerate(range(n-2, -1, -1)):
        var_to_find = elimination_order[i]
        # print(f'Finding range for variable x_{var_to_find}')
        # print(f'This can be achieved by substituting the following variables: ')
        vars_to_substitute = elimination_order[i+1:]
        # print('\t\t', ', '.join([f'x_{idx}' for idx in vars_to_substitute]))
        # print(f'In the system: ')
        system_to_subA, system_to_subB = history[i]
        # print(systos(system_to_subA, system_to_subB))
        new_sysA, new_sysB = system_to_subA, system_to_subB
        for subvar in vars_to_substitute:
            new_sysA, new_sysB = substitute(new_sysA, new_sysB, subvar, chosen_var_values[subvar])

        # print(f'That substitution gives a system:')
        # print(systos(new_sysA, new_sysB))


        found_interval = solve_single(new_sysA, new_sysB, var_to_find)

        if found_interval.empty:
            return None
        chosen_intervals[var_to_find] = found_interval
        # print(f'x_{var_to_find} is in {found_interval}, and we pick')
        if pick + 1 < len(value_picks):
            found_value = any_in_interval(found_interval) if value_picks == 'any' else value_picks[pick + 1]
            chosen_var_values[var_to_find] = found_value
            # print(f'x_{var_to_find} = {found_value}')

        # print('===================================')

    # print('-----------------------------------------------------------------------------')
    return chosen_intervals

# ============================ (2) POINT IN SYSTEM ===============================

# Checks wheter Ax >= b holds for given x
def point_in_system(A, b, x):
    A = np.array(A)
    b = np.array(b)
    x = np.array(x)
    return ((A @ x) >= b).all()



mA = [[7, 2, -2],
     [-1, -1, -1],
     [-2, 3, 1],
     [5, -1, 1]]

mb = [4, -4, 1, -2]

# ============================ (2) LINEAR PROGRAMMING ===============================
# Solves linear programming problem:
# (min) c^Tx
# where Ax <= b
# and x >= 0
def find_min(c, A, b):
    c = np.array(c, dtype=FLOAT_T)
    A = np.array(A, dtype=FLOAT_T)
    b = np.array(b, dtype=FLOAT_T)

    if misclose(c, 0.0).all():
        return 0.0

    # First we transform A and b in order to
    # flip the inequality sign, since Fourier-Motzkin works only with >=
    A = -A
    b = -b

    # Find k such that
    # c_k != 0 and
    # there is at least one inequality in the system
    # such that a_k != 0

    k_candidates = np.nonzero(np.invert(misclose(c, 0.0)))[0]
    if len(k_candidates) == 0:
        # This means that f is a constant (f = 0) and so that is the minimum
        return 0.0
    k = None
    done = False
    for row in A:
        for k_candidate in k_candidates:
            if not misclose(row[k_candidate], 0.0):
                k = k_candidate
                done = True
                break
        if done:
            break

    if k is None:
        # This means that no inequality contains any of the
        # variables in the function so basically I've no idea what to do
        # let's just say no solution
        return None

    # print(f'\tchoosing x_{k}')
    # Now we need to express x_k from the function, that is
    # f = c_0x_0 + .... c_kx_k + .... c_n-1x_n-1
    # f = [c0 c1 .... ck ..... cn-1]
    # ------
    # c_kx_k = -all_else + f
    # x_k = (1/c_k) (-all_else + f)
    # x_k = -all_else/c_k + f/ck
    # x_k = [-c_0/c_k, -c_1/c_k, .... 0 , -c_k+1/c_k  , ...., c_n-1/c_k, 1/c_k]

    x_k = np.hstack((-c/c[k], 1.0/c[k]))
    x_k[k] = 0.0
    # print(x_k)
    A = np.append(A, [[0.0] for _ in range(A.shape[0])], axis=1)
    m, n = A.shape
    # print(systos(A, b))
    # Now we insert x_k expression into every inequality
    for i, const in zip(range(m), b):
        if misclose(A[i][k], 0.0):
            continue
        A[i] = A[i] + A[i][k] * x_k
        A[i][k] = 0.0



    # Now just add all the x >= 0 inequalities:
    for i in range(n - 1):
        if i == k:
            # Here we don't add x_k >= 0 because x_k is removed
            # but we rather add the x_k expression to be greater than zero
            A = np.vstack((A, x_k))
            b = np.hstack((b, 0.0))
            continue
        cond = np.zeros(n)
        cond[i] = 1.0
        A = np.vstack((A, cond))
        b = np.hstack((b, 0.0))


    interval = fourier_motzkin(A, b, just_last=True)
    if interval is None:
        return None
    else:
        return interval.lower




if __name__ == '__main__':
    print('Working with system: ')
    print(systos(mA, mb))
    print('==============================================')

    print('1. Fourier-Motzkin, test1:')
    solution_intervals = fourier_motzkin(mA, mb, elimination_order=[2, 1, 0], value_picks=[1, 4])
    print('\r\n'.join(f'x_{i} is in {sol_interval}' for i, sol_interval in enumerate(solution_intervals))) if solution_intervals is not None else 'No solution.'

    print('==============================================')

    print('2. Point-in-system, test2:')
    p = [-1, -1, 1]
    in_str = 'is' if point_in_system(mA, mb, p) else 'is NOT'
    print(f'Point {p} {in_str} in the system.')

    print('==============================================')
    lpc = [-1, -3]
    lpA = [[1, 1],
        [-1, 2]]
    lpb = [6, 8]
    print('3. Linear programming, test3: ')
    print('Minimize f =', ' + '.join(f'{c}*x_{i}' for i, c in enumerate(lpc)))
    print('given:')
    print(systos(lpA, lpb))
    themin = np.round(find_min(lpc, lpA, lpb), 8)
    print(f'min(f)={themin}' if themin is not None else 'Unsolvable.')


    print('==============================================')
    print(f'4. Linear programming (poslednji primer iz sveske)')

    fA = [[1, -1, -1, 3],
        [5, 1, 3, 8],
        [-1, 2, 3, -5]]
    fb = [1, 55, 3]
    fc = [-4, -1, -5, -3]
    print(np.round(find_min(fc, fA, fb), 8))

