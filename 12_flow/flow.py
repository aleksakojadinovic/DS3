
import sys
import copy
import numpy as np
import pandas as pd
import argparse


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



            
def residual_nice(residual_network):
    
    V = len(residual_network)
    matrix =[['' for _ in range(V)] for _ in range(V)]
    
    sorted_keys = list(sorted([k for k in residual_network]))

    for i, node_from in enumerate(sorted_keys):
        for j, node_to in enumerate(sorted_keys):
            entry = residual_network[node_from][node_to]
            if entry is None:
                matrix[i][j] = 'N'
            else:
                flow, cap = entry
                matrix[i][j] = f'{flow} / {cap}'



    return pd.DataFrame(matrix, columns=sorted_keys, index=sorted_keys)
            


# Converts a network with just capacities into a network 
# where edge weight is x/y where x is current flow and y is capacity
# Also it adds residual edges
# Assumes full network ("dict matrix")
def network_to_residual_graph(network):
    residual_network = copy.deepcopy(network)
    # First pass to add forward edges
    for source_node in network:
        for target_node in network[source_node]:
            # We skip non existing edges for now
            if source_node == target_node or network[source_node][target_node] == 0:
                residual_network[source_node][target_node] = None
            else:
                # Otherwise we need to update its value to 0/capacity
                residual_network[source_node][target_node] = (0, network[source_node][target_node])


    
    n_residual_network = copy.deepcopy(residual_network)

    for node_from in residual_network:
        for node_to in residual_network:
            if node_from == node_to or residual_network[node_from][node_to] is None:
                continue
            # n_residual_network[node_to][node_from] = (-residual_network[node_from][node_to][1], 0)
            n_residual_network[node_to][node_from] = (0, 0)


    return n_residual_network
                

def get_temporary_bfs_graph(residual_network):
    nodes = [k for k in residual_network]
    tmp_graph = dict()
    for node in nodes:
        tmp_graph[node] = set()
    for node_from in nodes:
        for node_to in nodes:
            if node_from == node_to:
                continue
            if residual_network[node_from][node_to] is None:
                continue
            flow, cap = residual_network[node_from][node_to]
            if flow >= cap:
                continue
            tmp_graph[node_from].add(node_to)
    return tmp_graph

def reconstruct_path(start_node, end_node, parents):
    path = []
    current_node = end_node
    while True:
        path.append(current_node)
        if current_node == start_node:
            return list(reversed(path))
        current_node = parents[current_node]
        

def find_augmenting_path(residual_network, start_node='s', end_node='t'):
    graph = get_temporary_bfs_graph(residual_network)


    nodes = [k for k in graph]
    if start_node not in nodes:
        return None
    if end_node not in nodes:
        return None

    
    q = [start_node]
    visited = dict([(n, False) for n in nodes])
    parent = dict([(n, None) for n in nodes])

    while q:
        current_node = q.pop(0)
        if visited[current_node]:
            continue
        visited[current_node] = True
        if current_node == end_node:
            return reconstruct_path(start_node, end_node, parent)

        for next_node in graph[current_node]:
            if not visited[next_node] and next_node not in q:
                parent[next_node] = current_node
                q.append(next_node)

    return None


    


def strip_residuals(residual_network):
    nodes = [k for k in residual_network]
    new_network = copy.deepcopy(residual_network)

    for node_from in nodes:
        for node_to in nodes:
            if node_from == node_to or residual_network[node_from][node_to] is None:
                continue
                
            flow, cap = residual_network[node_from][node_to]

            if cap != 0:
                continue
            new_network[node_from][node_to] = None


    return new_network
    
    
def ek2(residual_network, source='s', sink='t'):
    iteration = 0
    while True:
        print(f'**** Iteration {iteration}')
        print(f'Current graph: ')
        print(residual_nice(residual_network))
        # if iteration == 1:
        #     print(f'force break for debug')
        #     break
        iteration += 1
        apath = find_augmenting_path(residual_network, source, sink)
        if apath is None:
            print(f'No path found.')
            break
        
        # Now find minimum slack and update edges along the path
        print(f'Found path {apath}')
        pairs = list(zip(apath, apath[1:]))
        flows = [residual_network[node_from][node_to] for node_from, node_to in pairs]
        slacks = [b - a for a, b in flows]
        # print(f'Edges : {pairs}')
        # print(f'Flows : {flows}')
        # print(f'Slacks: {slacks}')

        min_slack_idx = min(enumerate(slacks), key=lambda x:x[1])[0]
        min_slack = slacks[min_slack_idx]
        print(f'Minimum slack corresponds to edge {pairs[min_slack_idx]} and it is {min_slack}')

        
        # Now we update edges along the augmenting path by setting:
            # forward edges to flow + slack
            # backward edges to flow - slack
        
        for node_from, node_to in pairs:
            flow, cap = residual_network[node_from][node_to]
            new_flow = 0
            if cap == 0:
                new_flow = flow - min_slack
            else: 
                new_flow = flow + min_slack

            print(f'Updating edge {node_from} -- {node_to}')

            residual_network[node_from][node_to] = (new_flow, cap)

        # Also, if there is a forward edge on the path and its corresponding backward edge
        # is NOT in the path, we have to update that edge as well AND VICE VERSA
        for node_from, node_to in pairs:
            # If its reverse edge has already been used in path then it has been updared accordingly
            if (node_to, node_from) in pairs:
                continue

            # Now we know that this edge has been used in the path
            # but its reverse edge has not been used in the path

            # This seems counter intuitive (we use positive if its backwards)
            # That is because we're not updating this edge, but rather its corresponding reverse edge
            flow, cap = residual_network[node_from][node_to]
            to_add = min_slack if cap == 0 else -min_slack

            other_flow, other_cap = residual_network[node_to][node_from]
            residual_network[node_to][node_from] = (other_flow + to_add, other_cap)

    # Calculate flow
    # print(f'---- FINAL RESIDUAL GRAPH ---- ')
    # print(residual_nice(residual_network))

    # final_stripped = strip_residuals(residual_network)
    # print(f'Stripped residuals: ')
    # print(residual_nice(final_stripped))

    return



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

    residual_network = network_to_residual_graph(network)
    print(f'STARTED WITH: ')
    print(residual_nice(residual_network))


    ek2(residual_network)
    

    # if args.source not in nodes:
    #     print(f'Fatal error: Source node "{args.source}" not found.')
    #     sys.exit(1)
    # if args.end not in nodes:
    #     print(f'Fatal error: Sink node "{args.end}" not found.')
    #     sys.exit(1)
         

    # flow, capacities = edmonds_karp(network, nodes, s=args.source, t=args.end)
    # print(f'Max flow is {flow}')

    # og_matrix, og_df = dod_to_df(network)
    # ek_matrix, ek_df = dod_to_df(capacities)

    # print('OG: ')
    # print(og_df)

    # print(f'Capacities: ')
    # print(ek_df)

    

    




