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


def parse_input(filepath, style='adj_list'):
    if style not in ['adj_list', 'matrix']:
        raise ValueError(f'Unknown parse style: {style}')

    if style == 'matrix':
        raise NotImplementedError(f'Matrix parsing is not implemented yet!')

    return parse_adj_list(read_lines_ds(filepath))


        


        