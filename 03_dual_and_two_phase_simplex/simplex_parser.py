import numpy as np

def read_lines_ds(filepath):
    try:
        lines = open(filepath, "r").readlines()
        lines = (line.strip() for line in lines)
        lines = (line for line in lines if line)
        lines = (line for line in line if not line[0] == '#')
        lines = list(lines)
        return lines
    except:
        fatal_parse_error(f"Couldn't open file: {filepath}")

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


def auto_parse_constraints(lines):
    m = len(lines)
    lines = map(lambda line: line.split(" "), lines)
        
    
    
    