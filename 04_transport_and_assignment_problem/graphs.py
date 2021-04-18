import numpy as np
import utils as ut
import sys

def construct_graph(theta_i, theta_j, caps):
    m, n = caps.shape
    shape = caps.shape
    graph_matrix = [[] for _ in range(m*n)]
    
    def should_include_in_graph(ii, jj):
        return caps[ii][jj] == 1 or (ii, jj) == (theta_i, theta_j)

    for i in range(m):
        for j in range(n):
            if not should_include_in_graph(i, j):
                continue

            # Row edges
            for other_row in range(m):
                if not should_include_in_graph(other_row, j) or i == other_row:
                    continue
                # Edge between (i, j) and (other_row, j)
                graph_matrix[ut.pack_indices(i, j, shape)].append(ut.pack_indices(other_row, j, shape))


            # Column edges
            for other_col in range(n):
                if not should_include_in_graph(i, other_col) or j == other_col:
                    continue
                # Edge between (i, j) and (i, other_col)
                graph_matrix[ut.pack_indices(i, j, shape)].append(ut.pack_indices(i, other_col, shape))
                
    return graph_matrix

def graph_cleanup(graph_matrix):
    graph_dict = dict()
    for i, row in enumerate(graph_matrix):
        if row == []:
            continue
        graph_dict[i] = set(row)

    return graph_dict

def get_graph(i, j, caps):
    g = construct_graph(i, j, caps)
    g = graph_cleanup(g)
    return g

# TODO: This takes any valid cycle of length greater than 4.
#       Check whether it's enough or we need to find the longest one.
#       If so just add `candidates` list and argmax-it by length
# TODO: Also try to make it non-recursive as this shit is exponential af
def find_cycle(g, start_node, shape = None):
    all_nodes = list(g.keys())
    num_nodes = len(all_nodes)

    log_file = open("dfslog.txt", "w")
    stout_backup = sys.stdout
    sys.stdout = log_file

    def valid_branch_direction(last_direction, from_index, to_index):
        from_i, _ = ut.unpack_index(from_index, shape)
        to_i, _ = ut.unpack_index(to_index, shape)
        
        curr_direction = 1 if from_i == to_i else 0
        if last_direction == -1:
            return True, curr_direction

        
        # 1 means going horizontal
        # 0 means going vertical
        return last_direction != curr_direction, curr_direction

    def dfs(u, visited, path, last_direction=-1, depth=0):
        def print_depth(*args, **kwargs):
            print('\t'*depth, *args, **kwargs)

        print_depth(f'> At node {u}')
        visited = visited.copy()
        path = path.copy()
        visited[u] = True
        path.append(u)
        print_depth(f'\twith current path {path} and visited {[x for x, v in visited.items() if v]}')
        for neigh in g[u]:
            
                
            print_depth(f'\tChecking neighbor {neigh}')
            valid_move, curr_dir = valid_branch_direction(last_direction, u, neigh)
            if not valid_move:
                print_depth(f'\t\tInvalid move as last branch was also', 'horizontal' if last_direction == 1 else 'vertical')
                continue

            if neigh == start_node:
                print_depth(f'\t\tFound back edge to start node {start_node}.')
                if len(path) >= 4:
                    print_depth(f'\t\t\tLong enough!')
                    path.append(start_node)
                    return path
                else:
                    print_depth(f'\t\t\tToo short!')
            if visited[neigh]:
                print_depth(f'\t\tNeighbor already visited')
                continue
            
            print_depth(f'\t\tNeighbor not visited, going recursive.')
            potential_path = dfs(neigh, visited, path, last_direction=curr_dir, depth=depth+1)
            print_depth(f'\t\tBack at node {u}', "REMINDER: ", ' --> '.join(map(str, path)))
            if potential_path is not None:
                print_depth(f'\t\tNeighbor {neigh} found the path, done!')
                return potential_path
        
        return None
    res = dfs(start_node, dict((i, False) for i in all_nodes), [])
    sys.stdout = stout_backup
    return res
