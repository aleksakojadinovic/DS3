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
        fatal_parse_error(f'Expecting {n} {type_name}s, but {len(str_vals)} found., at {line}')
    
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


def parse_any_lp_input(lines):
    
    m, n = parse_matrix_dimensions(lines[0])

    c = parse_n_floats(n, lines[1])

    if len(lines[2:]) != m:
        fatal_parse_error(f'Expecting m rows, got {len(lines)}')

    

    eqA = []
    eqb = []
    leqA = []
    leqb = []
    for line in lines[2:]:
        line_sp = line.split(" ")
        # Expecting n + 2 values: n variables, 1 sign and 1 b value
        if len(line_sp) != n + 2:
            fatal_parse_error(f'Expecting {n+2} values in line {line} but got {len(line)}')

        sign = line_sp[-2]
        if sign not in ['=', '>=', '<=']:
            fatal_parse_error(f'Expecting sign to be >=, <= or =, but got {sign}')

        lhs = parse_n_floats(n, " ".join(line_sp[:n]))
        rhs = parse_n_floats(1, line_sp[-1])[0]

        if sign == '=':
            eqA.append(lhs)
            eqb.append(rhs)
        else:
            if sign == '>=':
                lhs = list(map(lambda x: -x, lhs))
                rhs = -rhs

            leqA.append(lhs)
            leqb.append(rhs)
    
    print(f'parsed:')
    print(c)
    print(eqA)
    print(eqb)
    print(leqA)
    print(leqb)

    return c, np.array(eqA), np.array(eqb), np.array(leqA), np.array(leqb)

def convert_to_eq(leqA, leqb, eqA=None, eqb=None):
    if len(leqA) == 0 or len(leqb) == 0:
        return eqA, eqb
    m, n = leqA.shape
    # We need to add m slack variables
    neweqA = np.zeros((m, n + m))

    neweqA[:m, :m] = leqA
    neweqA[m:, n:] = np.eye(m)
    neweqb = leqb

    if eqA is None or eqb is None or len(eqA) == 0 or len(eqb) == 0:
        return neweqA, neweqb

    

    other_m, other_n = eqA.shape
    if n > other_n:
        # This means that we have to extend other constraints
        diff = n - other_n
        eqA = np.append(eqA, np.zeros((other_m, diff)))
        eqb = np.append(eqb, np.zeros(diff))
    elif other_n > n:
        diff = other_n - n
        leqA = np.append(leqA, np.zeros((m, diff)))
        leqb = np.append(leqb, np.zeros(diff))

    return np.vstack(leqA, eqA), np.vstack(leqb, eqb)

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



