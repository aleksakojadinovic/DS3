import sys
from simple_graph import SimpleGraph

def fatal_parse_error(msg, line='unknown'):
    print(f'Fatal parse error at line {line}: {msg}')
    sys.exit(1)
    
def read_lines_ds(filepath):
    try:
        lines = open(filepath, "r").readlines()
        lines = enumerate(lines)
        lines = (line for line in lines if line[1][0] != '#')
        lines = ((i, line.strip()) for i, line in lines)
        lines = ((i, line) for i, line in lines if line)
        lines = list(lines)
        return lines
    except:
        fatal_parse_error(f"Couldn't open file: {filepath}")


"""
Input format:
V
i1 - i11, i12, ...
i2 - i21, i22, ...
...
ik - ik1, ik2, ...

"""
def parse_adj_list(lines):
    V = None
    try:
        V = int(lines[0][1])
    except:
        fatal_parse_error(f'Failed to parse number of nodes from {lines[0][1]}', line=0)

    g = SimpleGraph(V)

    for line_number, line in lines[1:]:
        src_dist_strings = line.split("-")
        if "-" not in line or len(src_dist_strings) != 2:
            fatal_parse_error(f"Couldn't parse line: {line}, expecting format: source_node - dest1, dest2, ...", line=line_number)
        
        src_string = src_dist_strings[0]
        dst_string = src_dist_strings[1]

        src = None
        dests = None
        try:
            src = int(src_string)
        except:
            fatal_parse_error(f"Non-integer source node: {src_string}", line=line_number)

        try:
            dests = list(map(int, dst_string.split(",")))
        except:
            fatal_parse_error(f"Non-integer(s) found in destination nodes: {dst_string}", line=line_number)
        

        for dest in dests:
            g.connect(src, dest)

    return g

def parse_row(line):
    i, row = line
    row = row.split(" ")
    try:
        row = list(map(int, row))
        return row
    except:
        fatal_parse_error(f'Failed to parse {row}, ints expected.', line=i)

def parse_matrix(lines):
    if len(lines) <= 0:
        fatal_parse_error(f'Zero length matrix.')

    row0 = parse_row(lines[0])
    n = len(row0)
    if len(lines) != n:
        fatal_parse_error(f'Expecting square matrix but the matrix given is {len(lines)} x {n}', line=0)

    matrix = []

    for i, line in lines:
        row = parse_row((i, line))
        if any(map(lambda x: x != 0 and x != 1)):
            fatal_parse_error(f'Expecting binary matrix.', line=line)

        matrix.append(row)

    g = SimpleGraph(n)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                g.connect(i, j)

    return g


        
        
    


def parse_input(filepath, style='adj_list'):
    if style not in ['adj_list', 'matrix']:
        fatal_parse_error(f'Unknown input style {style}')

    if style == 'matrix':
        return parse_matrix(read_lines_ds(filepath))

    if style == 'adj_list':
        return parse_adj_list(read_lines_ds(filepath))

    


        


        