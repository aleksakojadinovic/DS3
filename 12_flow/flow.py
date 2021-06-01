from os import name
from types import new_class

import sys
import copy
import numpy as np
import pandas as pd
import argparse


from collections import defaultdict

def fatal(msg):
    print(f'Fatal parse error: {msg}')
    sys.exit(1)

def read_lines_ds(filepath):
    try:
        lines = open(filepath, "r").readlines()
        lines = (line.strip() for line in lines)
        lines = (' '.join(line.split()) for line in lines)
        lines = (line for line in lines if line)
        lines = (line for line in lines if line[0] != '#')
        lines = list(lines)
        return lines
    except:
        fatal(f'Could not open graph file: {filepath}')

def parse_flow_network(filepath):
    lines = read_lines_ds(filepath)
    # V = None
    # try:
    #     V = int(lines[0])
    # except:
    #     fatal(f'Cannot parse number of nodes from {lines[0]}')

    network = dict()
    node_markers = set()
    for line in lines:
        line_strings = line.split(" ")
        if len(line_strings) != 3:
            fatal(f'Expecting three entries per line, got {line}')
        source      = line_strings[0]
        dest        = line_strings[1]
        cap_string  = line_strings[2]

        node_markers.add(source)
        node_markers.add(dest)

        cap_val = None
        try:
            cap_val = int(cap_string)
        except:
            fatal(f'Failed to parse capacity from {cap_string}')
        
        if source not in network:
            network[source] = dict()
        network[source][dest] = cap_val


    for node_from in node_markers:
        if node_from not in network:
            network[node_from] = dict()
        for node_to in node_markers:
            if node_from == node_to or node_to not in network[node_from]:
                network[node_from][node_to] = 0




    return network, node_markers
        


def ek_bfs(capacities, start_node, end_node):
    ret = 0

    parents = dict([(node, None) for node in capacities])

    qnodes      = [start_node]
    qcapacities = [float('inf')]

    while qnodes:
        curr_node = qnodes.pop(0)
        capacity = qcapacities.pop(0)

        for neighbor_node in capacities[curr_node]:
            if capacities[curr_node][neighbor_node] == 0:
                continue
            if capacities[curr_node][neighbor_node] > 0 and parents[neighbor_node] is None:
                qnodes.append(neighbor_node)
                qcapacities.append(min(capacity, capacities[curr_node][neighbor_node]))
                parents[neighbor_node] = curr_node

            if neighbor_node == end_node:
                ret = min(capacity, capacities[curr_node][neighbor_node])
                break
    
    

    if ret > 0:
        curr_node = end_node
        while curr_node != start_node:
            capacities[parents[curr_node]][curr_node] -= ret
            capacities[curr_node][parents[curr_node]] += ret
            curr_node = parents[curr_node]


    return ret


    

def edmonds_karp(network, nodes, s='s', t='t'):
    network = copy.deepcopy(network)
    if s not in nodes:
        raise ValueError(f'Invalid source node: {s}')
    if t not in nodes:
        raise ValueError(f'Invalid sink node: {t}')

    max_flow = 0
            
    while True:
        current_flow = ek_bfs(network, s, t)
        if current_flow == 0:
            break
        max_flow += current_flow
    
    return max_flow, network

def dod_to_df(dod: dict):
    keys = dod.keys()
    keys = list(keys)
    keys = list(sorted(keys))
    matrix = np.zeros((len(keys), len(keys)), dtype='int32')

    for i, key in enumerate(dod):
        entries = [(k, dod[key][k]) for k in dod[key]]
        entries = list(sorted(entries, key=lambda e: e[0]))
        matrix[i] = list(map(lambda x: x[1], entries))


    df = pd.DataFrame(matrix, index=keys, columns=keys)
    return matrix, df




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input',
                        required=True,
                        help='The input file.')

    parser.add_argument('-v',
                        '--visualize',
                        action='store_true',
                        help='Visualize the resulting tree.')

    parser.add_argument('-p',
                        '--prog',
                        default='neato',
                        help='Which graphviz visualization program to use.')

    parser.add_argument('-s',
                        '--source',
                        default='s',
                        help='The source node.')

    parser.add_argument('-e',
                        '--end',
                        default='t',
                        help='The sink node.')

    args = parser.parse_args()
    network, nodes = parse_flow_network(args.input)

    if args.source not in nodes:
        print(f'Fatal error: Source node "{args.source}" not found.')
        sys.exit(1)
    if args.end not in nodes:
        print(f'Fatal error: Sink node "{args.end}" not found.')
        sys.exit(1)
         

    flow, capacities = edmonds_karp(network, nodes, s=args.source, t=args.end)
    print(f'Max flow is {flow}')

    og_matrix, og_df = dod_to_df(network)
    ek_matrix, ek_df = dod_to_df(capacities)

    print('OG: ')
    print(og_df)

    print(f'Capacities: ')
    print(ek_df)

    

    




