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
a00   a01    .... a0n-1     >=/<= b0
a10   a11    .... a1n-1     >=/<= b1
...
am-10 am-11  .... am-1n-1   >=/<= bm

and returns it in Ax <= b form
"""
def parse_constraint_matrix(m, n, lines):
    try:
        if len(lines) != m:
            fatal_parse_error(f'Expected {m} lines in constraint matrix, got {len(lines)}')
        A = np.zeros((m, n), dtype='float32')
        b = np.zeros(m, dtype='float32')
        for i, row_string in enumerate(lines):
            row_strings = row_string.split(" ")
            if len(row_strings) != n + 2:
                fatal_parse_error(f'Row {row_string} should contain {n + 2} values ({n} variables, 1 sign and 1 coefficient) but it contains {len(row_strings)}')
            sign = row_strings[-2]
            if sign != '>=' and sign != '<=':
                fatal_parse_error(f'Unknown sign: {sign}, expecting >= or <=')
            
            a_vals = list(map(float, row_strings[:-2]))
            b_val  = float(row_strings[-1])

            A[i] = np.array(a_vals) if sign == '<=' else -np.array(a_vals)
            b[i] = b_val if sign == '<=' else -b_val

        return A, b
    except:
        fatal_parse_error(f'Failed to parse constraint matrix (probably non-numbers found).')

"""
Expects the entire problem in canonical for, meaning the entire simplex matrix:
a00     a01    ........ a0n-1    b0
a10     a11    ........ a1n-1    b1
               ........ 
am-10   am-11           am-1n-1  bm-1
c0        c1             cn-1     0
"""
def parse_full_canonical(lines):
    return parse_mn_matrix(len(lines), len(lines[0].split(" ")), lines)


# NOTE: The following two functions represent the only two ways that the 
# simplex solver will accept the input. Other parsers may be written
# completely manually, or by using some of the given functions here.

"""
Expects the following form:
m n
-- target function coeffs --
-- constraint matrix with >= and <= signs

and returns A b c form
can further be prepared for simplex with abc_to_simplex_matrix
"""
def parse_full_constraint(lines):
    m, n = parse_matrix_dimensions(lines[0])
    c    = parse_n_floats(n, lines[1])
    A, b = parse_constraint_matrix(m, n, lines[2:])

    return A, b, c

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



"""
Takes in filepath and flags provided through the command line
Returns the simplex matrix for the algorithm
All algorithms will work with the standard canonical form
so this function takes care of every possible flag
"""
def prepare_for_algorithm(run_flags):
    lines = read_lines_ds(run_flags['input'])
    if run_flags['format'] == 'constraints':
        A, b, c = parse_full_constraint(lines)
        simplex_matrix = abc_to_simplex_matrix(A, b, c)
    elif run_flags['format'] == 'simplex_matrix':
        simplex_matrix = parse_full_canonical(lines)
    else:
        print(f"Unknown format: {run_flags['format']}, expecting either 'constraints' or 'simplex_matrix'")
    
    if run_flags['maximize']:
        simplex_matrix[:-1] *= -1

    # Also returns run_flags because info about
    # Blend rule and possibly something else is there
    return simplex_matrix, run_flags

def get_simplex_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', 
                        '--input',
                         help='The input file',
                         required=True)
    parser.add_argument('-m',
                        '--maximize',
                        action='store_true',
                        help='Maximize instead of minimize')

    parser.add_argument('-p',
                        '--printproblem',
                        action='store_true',
                        help='Print a human readable representation of the problem first')
    
    parser.add_argument('-f',
                        '--format',
                        default='simplex_matrix',
                        help='Whether to provide simplex matrix (`simplex_matrix`) or Abc form (`constraints`)')

    parser.add_argument('-l',
                        '--logging',
                        action='store_true',
                        help='Display log messages.')

    return parser


def print_solution(run_flags, x, val):
    target = 'Max' if run_flags['maximize'] else 'Min'
    if run_flags['logging']:
        print('<<<<<<<<<<SOLUTION>>>>>>>>>>')
    print(f'{target} point: {x}')
    print(f'{target} value: {val}')
