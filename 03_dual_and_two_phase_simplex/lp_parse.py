"""Parses input for linear constraint problem"""
import numpy as np
import sys
import argparse

def fatal_parse_error(msg):
    print(f'Fatal parse error: {msg}')
    sys.exit(1)
    
def read_lines_ds(filepath):
    try:
        lines = open(filepath, "r").readlines()
        lines = (line.strip() for line in lines)
        lines = (line for line in lines if line)
        lines = (line for line in lines if line[0] != '#')
        lines = list(lines)
        return lines
    except:
        fatal_parse_error(f"Couldn't open file: {filepath}")

def parse_n_of(n, line, type_func, type_name):
    if n <= 0:
        fatal_parse_error(f'Expecting at least one value, got {n}')
    str_vals = line.split(" ")
    if len(str_vals) != n:
        fatal_parse_error(f'Expecting {n} {type_name}s, but {len(str_vals)} found.')
    
    try:
        res = np.array(list(map(type_func, str_vals)))
        return res
    except:
        fatal_parse_error(f'Failed to convert some values to {type_name}.')

def parse_n_ints(n, line):
    return parse_n_of(n, line, int, 'integer')

def parse_n_floats(n, line):
    return parse_n_of(n, line, float, 'float')

def parse_matrix_dimensions(line):
    try:
        vals = line.split(" ")
        if len(vals) != 2:
            fatal_parse_error(f'Expecting two values for dimensions, got {len(vals)}.')
        m = int(vals[0])
        n = int(vals[1])
        return m, n
    except:
        fatal_parse_error(f'Failed to parse matrix dimensions from: {line}.')

def parse_mn_matrix(m, n, lines):
    if m <= 0 or n <= 0:
        fatal_parse_error(f'Invalid dimensions: {m} x {n}.')
    if len(lines) != m: 
        fatal_parse_error(f'Expecting {m} lines, got {len(lines)}.')
    matrix = np.zeros((m, n), dtype='float32')
    for i, row_string in enumerate(lines):
        row = parse_n_floats(n, row_string)
        matrix[i] = row
    return matrix

"""Parses matrix of form:
a00   a01    .... a0n-1     >=/<=/= b0
a10   a11    .... a1n-1     >=/<=/= b1
...
am-10 am-11  .... am-1n-1   >=/<=/= bm

and returns it in eqA, eqb, leqA, leqb form (the simplex format)
"""
def parse_any_lp_input(lines):
    m, n = parse_matrix_dimensions(lines[0])
    lines = lines[1:]
    if len(lines) != m:
        fatal_parse_error(f'Expecting m rows, got {len(lines)}')

    eqA = []
    eqb = []
    leqA = []
    leqb = []
    for line in lines[1:]:
        line = line.split(" ")
        # Expecting n + 2 values: n variables, 1 sign and 1 b value
        if len(line) != n + 2:
            fatal_parse_error(f'Expecting {n+2} values in line {line} but got {len(line)}')

        sign = line[-2]
        if sign not in ['=', '>=', '<=']:
            fatal_parse_error(f'Expecting sign to be >=, <= or =, but got {sign}')

        lhs = parse_n_floats(n, line[:n])
        rhs = parse_n_floats(1, [line[-1]])[0]

        if sign == '=':
            eqA.append(lhs)
            eqb.append(rhs)
        else:
            if sign == '>=':
                lhs = list(map(lambda x: -x, lhs))
                rhs = -rhs

            leqA.append(lhs)
            leqb.append(rhs)
    

    return eqA, eqb, leqA, leqb

        

    
    


def abc_to_simplex_matrix(A, b, c):
    A = np.array(A)
    b = np.array(b)
    c = np.array(c)

    m, n = A.shape
    simplex_matrix = np.zeros((m + 1, n + m + 1))
    simplex_matrix[:m, :n] = A
    simplex_matrix[:-1, n:-1] = np.eye(m)
    simplex_matrix[-1, :] = np.append(c, np.zeros(m + 1))
    simplex_matrix[:-1, -1] = b
    return simplex_matrix

def simplex_matrix_to_abc(simplex_matrix):
    # NOTE: Not possible if malformed
    simplex_m, simplex_n = simplex_matrix.shape
    m = simplex_m - 1
    n = simplex_n - m - 1
    A = simplex_matrix[:m, :n]
    b = simplex_matrix[:-1, -1]
    c = simplex_matrix[-1, :]
    if (c[n:] != 0).any():
        raise ValueError(f'Target function has been modified, cannot convert.')
    return A, b, c[:n]



