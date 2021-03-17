import numpy as np
import portion
import sys
import argparse
import time

LOG =  False
DEC = 8

FILE_FORMAT_STRING ='\r\n'.join(['M N',
                                'c1 c2 ... cn',
                                'a11 a12 ... a1n b1',
                                'a21 a22 ... a2n b2',
                                ' ... ',
                                'am1 am2 ... amn bm'])

def log(*args, **kwargs):
    if LOG:
        print(*args, **kwargs)


"""Linear expression to string"""
def lexp_to_string(exp):
    return ' + '.join(f'{c}*x_{i}' for i, c in enumerate(exp))

def lineq_to_string(a, b, sign):
    return lexp_to_string(a) + ' ' + sign + ' ' + str(b)

def print_linear_programming_problem(A, b, c, is_max, is_greater):
    sign = '>=' if is_greater else '<='
    target_function_string = lexp_to_string(c)
    constraints_string = '\r\n'.join(lineq_to_string(a, b, sign) for a, b in zip(A, b))

    print('<<< PROBLEM >>>')
    print('Maximize:' if is_max else 'Minimize:')
    print(f'f = {target_function_string}')
    print('Subject to constraints:')
    print(constraints_string)
    print('x_i >= 0, for all i')
    print('===============')


"""Returns a tuple (simplex_matrix, Q, P, x0)"""
def to_canonical(A, b, c):
    # TODO: Optimize by allocating entire matrix once and then filling values with indexing
    A = np.array(A)
    b = np.array(b)
    c = np.array(c)
    m, n = A.shape

    s_matrix = np.append(A, np.eye(m), axis=1)
    s_matrix = np.vstack((s_matrix, np.append(c, np.zeros(s_matrix.shape[1] - n))))
    s_matrix = np.hstack((s_matrix, np.append(b, 0).reshape(-1, 1)))
    P = np.array(range(n, s_matrix.shape[1] - 1))
    Q = np.array(range(n))

    return s_matrix, Q, P, np.append(np.zeros(s_matrix.shape[1] - len(b) - 1), b)



def canonical_simplex(simplex_matrix, Q, P, x0, flags):
    eta = flags['eta']
    if eta:
        log('>> Starting Revised Simplex Algorithm.')
    else:
        log('>> Starting Revised Simplex Algorithm with ETA matrices.')
    
    iteration = 1
    
    simplex_m, simplex_n = simplex_matrix.shape

    b = simplex_matrix[:-1, -1]
    c = simplex_matrix[-1:, :].flatten()

    current_basis_matrix = None
    if eta:
        current_basis_matrix = np.eye(len(P))

    while True:
        log(f'****** ITERATION {iteration}')
        iteration += 1

        log(f'Data: P={P}, Q={Q}, x0={x0}')

        if eta:
            log('Basis matrix: ')
            log(current_basis_matrix)

        log('--------------------------------------------------------')

        if eta:
            u = np.linalg.solve(current_basis_matrix.T, c[P])
        else:
            u = np.linalg.solve(simplex_matrix[:-1, P].T, c[P])

        log(f'u = {u}')

        pure_f = np.zeros(simplex_n)
        pure_f[-1] = np.dot(u, b)
        
        for i in range(len(pure_f) - 1):
            pure_f[i] = c[i] - np.dot(u, simplex_matrix[:-1, i]) if i in Q else 0
        
        if (pure_f[:-1] >= 0).all():
            return np.around(x0, DEC)[:-simplex_m + 1], np.round(np.dot(x0, c[:-1]), DEC)

        j = np.where(pure_f[:-1] < 0)[0][0]        
        if eta:
            y_sol = np.linalg.solve(current_basis_matrix, simplex_matrix[:-1, j])
        else:
            y_sol = np.linalg.solve(np.array(simplex_matrix[:-1, P]), simplex_matrix[:-1, j])
        y = np.zeros(simplex_n)
        y[P] = y_sol
        log(f'y = {y_sol}')

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
            return None, None
        t = t_interval.upper
        log(f't = {t}')

        s = None
        for s_candidate in P:
            if y[s_candidate] <= 0:
                continue
            if x0[s_candidate] - t*y[s_candidate] == 0:
                s = s_candidate
                break
        
        if s is None:
            return None, None

        log(f's = {s}')

        x1 = np.zeros(len(x0))
        x1[j] = t
        for i in P[P != s]:
            x1[i] = x0[i] - t * y[i]
        

        newQ = np.append(Q[Q != j], s)
        newP = np.where(P == s, j, P)

        if eta:
            eta_matrix = np.eye(len(P))
            actual_s_index = np.where(P == s)[0][0]
            eta_matrix[:, actual_s_index] = y_sol
            log('eta=')
            log(eta_matrix)                      
            current_basis_matrix = current_basis_matrix @ eta_matrix

        x0 = x1
        P = newP
        Q = newQ


def parse_error(msg=''):
    print(f'Error parsing input file: {msg}\r\n',
            'NOTE:',
            FILE_FORMAT_STRING,
            sep='\r\n'
            )

def parse_dimensions(dim_line):
    dims_strings = dim_line.split(' ')
    if len(dims_strings) != 2:
        parse_error(f'Expected two integers in the first line, got `{dim_line}`')

    try:
        m = int(dims_strings[0])
        n = int(dims_strings[1])
        return m, n
    except:
        parse_error(f'Expected two integers in the first line, got {dim_line}')

def parse_target_function(target_function_line, n):
    target_coeffs_strings = target_function_line.split(' ')
    if len(target_coeffs_strings) != n:
        parse_error(f'Expecting {n} target coefficients, got {len(target_coeffs_strings)}')
    try:
        coeffs = list(map(float, target_coeffs_strings))
        return np.array(coeffs)
    except:
        parse_error(f'One or more of target coefficients is not a float.')
    
def parse_constraint_matrix(constraint_matrix_lines, m, n):
    if len(constraint_matrix_lines) != m:
        parse_error(f'Expecting {m} lines in constraint matrix, got {len(constraint_matrix_lines)}')
    matrix = np.zeros((m, n))
    b_vector = np.zeros(m)
    for i in range(m):
        row = constraint_matrix_lines[i].split(' ')
        if len(row) != n + 1:
            parse_error(f'Expecting {n + 1} values per row in constraint matrix, but row {row} has {len(row)}')
        for j, str_val in enumerate(row):
            try:
                float_val = float(str_val)
                if j == n:
                    b_vector[i] = float_val
                else:
                    matrix[i][j] = float_val
            except:
                parse_error(f'Non float in constraint matrix: {str_val}')
    
    return matrix, b_vector

def parse_input(input_lines):
    if len(input_lines) < 3:
        parse_error(f'Expected 3 lines, got {len(input_lines)}')
    
    m, n = parse_dimensions(input_lines[0])
    c = parse_target_function(input_lines[1], n)
    A, b = parse_constraint_matrix(input_lines[2:], m, n)
    return A, b, c

def fetch_input(file_name):
    try:
        input_file = open(file_name, "r")
        lines = input_file.readlines()
        lines = (line.strip() for line in lines)
        lines = (line for line in lines if line)
        lines = list(lines)
        return parse_input(lines) 
    except:
        print('Failed to open input file.')
        sys.exit(1)


parser = argparse.ArgumentParser()

parser.add_argument('--format',
                    action='store_true',
                    help='Show the input file format')

parser.add_argument('-i',
                    '--input',
                    help='The input file')

parser.add_argument('-e', 
                    '--eta',
                     action='store_true',
                     help='Whether to use ETA matrices')


parser.add_argument('-m',
                    '--max',
                    action='store_true',
                    help='Maximize instead of minimize')


parser.add_argument('-g',
                    '--greater',
                    action='store_true',
                    help='Use >= instead of <=')

parser.add_argument('-p',
                    '--printproblem',
                    action='store_true',
                    help='Print a human readable representation of the problem first')

parser.add_argument('-l',
                    '--logging',
                    action='store_true',
                    help='Print log messages throughout the algorithm')





args = vars(parser.parse_args())

if args['format']:
    print(FILE_FORMAT_STRING)
    sys.exit(0)

LOG = args['logging']

if args['input'] is None:
    print('No input file specified.')
    sys.exit(1)


A, b, c = fetch_input(args['input'])

if args['printproblem']:
    print_linear_programming_problem(A, b, c, args['max'], args['greater'])

if args['max']:
    c *= -1

if args['greater']:
    A *= -1
    b *= -1


simplex_matrix, Q, P, x0 = to_canonical(A, b, c)
ex_x, ex_f = canonical_simplex(simplex_matrix, Q, P, x0, args)
ex_f = -1*ex_f if args['max'] else ex_f

if LOG:
    print('=================================================')
if ex_x is None:
    print(f'The function is unbounded (no solution).')
else:
    print('Maximum point' if args['max'] else 'Minimum point' , ':', ex_x)
    print(f'Maximum value' if args['max'] else 'Minimum value', ':', ex_f)

