import copy 
import argparse
from simple_graph import SimpleGraph
import graph_parser as gp


def get_euler_start_node(graph):
    ec = graph.euler_check()
    if ec == SimpleGraph.EULER_HAS_NOTHING:
        return None

    if ec == SimpleGraph.EULER_HAS_CYCLE:
        # We can start anywhere
        return 0
    else:
        # We have to start at an odd node
        return [i for i, deg in enumerate(graph.degrees) if deg % 2 == 1][0]

# Picks an edge from node for Fleury's algorithm
def pick_edge(graph, node):
    candidate_edges = graph.adj_list[node]
    if len(candidate_edges) == 1:
        return list(candidate_edges)[0]

    for neighbor in candidate_edges:
        # Check whether this edge is a bridge
        tmp_graph = copy.deepcopy(graph)
        tmp_graph.disconnect(node, neighbor)
        visited = tmp_graph.dfs(start_node=node)
        if visited[neighbor]:
            return neighbor


    raise ValueError(f'Could not find appropriate edge!')

def fleury_algorithm(graph):
    start_node = get_euler_start_node(graph)
    if start_node is None:
        return None

    graph = copy.deepcopy(graph)

    V = graph.num_nodes
    E = graph.num_edges
    curr_path = [start_node]
    curr_node = start_node

    edges_hit = 0

    while True:

        if edges_hit == E:
            break
        
        node_to = pick_edge(graph, curr_node)
        graph.disconnect(curr_node, node_to)
        curr_path.append(node_to)
        curr_node = node_to
        edges_hit += 1

    return curr_path




def example1():
    g = SimpleGraph(4)
    g.connect(0, 1)
    g.connect(0, 2)
    g.connect(1, 2)
    g.connect(2, 3)
    print(fleury_algorithm(g))

def example2():
    g = SimpleGraph(5)
    g.connect(0, 1)
    g.connect(1, 2)
    g.connect(2, 0)
    print(fleury_algorithm(g))

def example3():
    g = SimpleGraph(5)
    g.connect(1, 0)
    g.connect(0, 2)
    g.connect(2, 1)
    g.connect(0, 3)
    g.connect(3, 4)
    g.connect(3, 2)
    g.connect(3, 1)
    g.connect(2, 4)
    print(fleury_algorithm(g))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', 
                        '--input',
                         help='The input file',
                         required=True)
    

    parser.add_argument('-s',
                        '--style',
                        required=True,
                        default='adj_list',
                        help='Input style, either `adj_list` or `matrix`.')


    args = parser.parse_args()
    g = gp.parse_input(args.input, style=args.style)    
    print(fleury_algorithm(g))



    
        
