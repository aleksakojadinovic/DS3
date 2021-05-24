import sys

import numpy as np
import argparse

from typing import List

from numpy.core.numeric import outer

from graphs import DirectedGraph
from graphs import bfs

import networkx as nx
import matplotlib.pyplot as plt

        
STDOUT = sys.stdout
LOGFILE = 'out.txt'
sys.stdout = open(LOGFILE, 'w')

def update_edge_list(edge_list: List[tuple[int, int, float]], supernode: int, nodes_in_cycle: List[int], edges_in_cycle: List[tuple[int, int, float]]):
    edge_list = list(filter(lambda edge: edge[0] not in nodes_in_cycle or edge[1] not in nodes_in_cycle, edge_list))

    lost_outgoing_edges = set()

    for edge_idx, (u, v, w) in enumerate(edge_list):
        if u in nodes_in_cycle:
            lost_outgoing_edges.add((u, v, w))
            edge_list[edge_idx] = (supernode, v, w)
            # But I need to memorize that there used to be an edge from u to v
            

    lost_incoming_edges = set()

    for edge_idx, (u, v, w) in enumerate(edge_list):
        if u not in nodes_in_cycle and v in nodes_in_cycle:
            # This should now be replaced with an edge (u, e_start_node) with weight w - w(fc) where fc is in the cycle and its destination is v
            replacement_candidates = list(filter(lambda edge: edge[1] == v, edges_in_cycle))
            if not replacement_candidates:
                sys.exit(1)


            lost_incoming_edges.add((u, v, w))

            fcu, fcv, fcw = replacement_candidates[0]
            edge_list[edge_idx] = (u, supernode, w - fcw)
    
    
    return edge_list, lost_incoming_edges, lost_outgoing_edges

def pick_root(graph: DirectedGraph):
    # Nodes with no incoming edges
    no_in_nodes = list(filter(lambda entry: entry[1] == 0, enumerate(graph.in_degrees)))
    if len(no_in_nodes) > 1:
        return -1, 'Cannot pick root, there are {len(no_in_nodes)} nodes with no incoming edges.'
    
    # If there is a node with no incoming edges then it has to be root
    if no_in_nodes:
        return no_in_nodes[0][0], ''

    # Otherwise choose a node with max out edges
    max_out_degree_entry = max(enumerate(graph.out_degrees), key=lambda e: e[1])
    return max_out_degree_entry[0], ''


def edmonds(graph: DirectedGraph, r: any = 'auto') -> None:

    
    # Step 0
    if r == 'auto' or r is None:
        # We're choosing highest out-degree node
        # max_out_degree_entry = max(enumerate(graph.out_degrees), key=lambda e: e[1])
        # r = max_out_degree_entry[0]
        r, msg = pick_root(graph)
        if r is None:
            print(msg)
            sys.exit(1)

        print(f'Chosen root {r}')

    if r < 0 or r >= graph.num_nodes:
        raise ValueError(f'Root {r} out of bounds for graph with {graph.num_nodes} vertices.')    


    # Step 1
    V = [i for i in range(graph.num_nodes)]
    E = graph.edges()
    W = [r]
    F = []
    
    super_nodes_ = []
    all_covered_nodes_ = set()

    bfs_numbers = bfs(graph, start=r)


    print(f'Starting Edmonson algorithm.')

    outer_iter = 0
    while True:
        print(f'>> Outer iteration {outer_iter} with the following state: ')
        print(f'\t\t graph edges E= {E}')
        print(f'\t\t new nodes   W= {W}')
        print(f'\t\t new edges   F= {F}')
        print(f'\t\t all_covered_nodes: {all_covered_nodes_}')
        
        if set(W).union(all_covered_nodes_) == set(V):
            # Step 8:
            break
        # Step 3: Find node x not in W with max BFS number
        bfs_result = bfs(graph, start=r)
        print(f'BFS numbers: {bfs_result["numbers"]}')
        max_entry = max(filter(lambda entry: entry[0] not in W and entry[0] not in all_covered_nodes_, enumerate(bfs_result["numbers"])), key=lambda entry: entry[1])
        x = max_entry[0]
        print(f'Node with max number is node {x}')
        print(f'Initializing path to empty')
        Vp = []
        Ep = []
        
        inner_iter = 0
        while True:
            print(f'\t ** Inner iteration {inner_iter} started with the following state:')
            print(f'\t\t\t graph edges E= {E}')
            print(f'\t\t\t new nodes   W= {W}')
            print(f'\t\t\t new edges   F= {F}')
            print(f'\t\t\t all_covered_nodes: {all_covered_nodes_}')
            print(f'\t\t\t X = {x}')
            print(f'\t\t\t current_path_nodes Vp= {Vp}')
            print(f'\t\t\t current_path_edges Ep= {Ep}')

            # Step 4: Form a path P such that V(P) := V(P) U {x} and E(P) := E(P) U {e} where e is the minimum weight edge that ends in x
            Vp.append(x)
            x_input_edges = list(filter(lambda edge: edge[1] == x, E))
            if not x_input_edges:
                print(f'Minimum edge not found, fuck this.')
                sys.exit(1)
            e = min(x_input_edges, key=lambda edge: edge[2])
            print(f'\t\tMinimum input edge into node x={x} is edge {e}')
            Ep.append(e)

            e_start_node = e[0]
            # Step 5: If the start of edge e is not in Vp U W set it to x and return to step 4
            if e_start_node not in set(Vp).union(set(W)):
                print(f'\t\tThis edge starts in node {e_start_node} which is not in Vp U W, therefore we put it as x and continue finding path.')
                x = e_start_node
                inner_iter += 1
                continue

            # Step 6: If the start of edge e is in Vp then we have a cycle
            if e_start_node in Vp:
                print(f"\t\tThis edge starts in node {e_start_node} which is already in our current path, therefore we've got a cycle!.")
                # We have a cycle C (in subgraph Vp, Ep)
                # BUT The cycle may only consist of subset of these nodes
                previous_appearence_idx = Vp.index(e_start_node)
                nodes_in_cycle = Vp[previous_appearence_idx:]
                edges_in_cycle = [(u, v, w) for (u, v, w) in Ep if u in nodes_in_cycle or v in nodes_in_cycle]
                print(f"\t\t\t The cycle consist of the nodes {nodes_in_cycle}")
                print(f"\t\t\t The cycle consist of the edges {edges_in_cycle}")
                
                

                super_nodes_.append({
                    "node_idx": e_start_node,
                    "covers_nodes": list(nodes_in_cycle),
                    "cycle_edges": list(edges_in_cycle),
                    "non_edge": e
                })
                removed_nodes = list(n for n in nodes_in_cycle if n != e_start_node)

                print(f"\t\t\t We introduce {e_start_node} as super node that covers nodes {removed_nodes}")

                all_covered_nodes_ = all_covered_nodes_.union(set(removed_nodes))
                # Now we must modify Vp and Ep
                # We remove all `removed_nodes` from Vp
                Vp = list(filter(lambda x: x not in removed_nodes, Vp))
                print(f"\t\t\t We remove all covered nodes from current Vp getting {Vp}")

                
                # Now, this also effects (I guess) both the Ep edges and the E edges, as other edges are used to construct path
                # Therefore we must apply the same edge_replace procedure to them
                # It states as follows:
                    # For all edges f that start outside of C and end in C (edges in edges_in_cycle definitely start within cycle so we dont care about them), replace their destinations with supernode c
                        # Also modify their weights to we w(f) := w(f) - w(fC) where fC is the input edge to the cycle C 
                        # that has the same destination as edge f

                # These affected edges are either edges in Ep that did not participate in the cycle or edges in total graph (E) that are to be added to graph later
                print(f"\t\t\t We now need to update path edges that are currently: {Ep}")
                Ep, lost_inc1, lost_out1 = update_edge_list(Ep, e_start_node, nodes_in_cycle, edges_in_cycle)
                print(f"\t\t\t After update we have Ep={Ep}")

                print(f"\t\t\t We now need to update global edges: {E}")
                E, lost_inc2, lost_out2 = update_edge_list(E, e_start_node, nodes_in_cycle, edges_in_cycle)
                print(f"\t\t\t After update we have E={E}")

                lost_inc = lost_inc1.union(lost_inc2)
                lost_out = lost_out1.union(lost_out2)

                super_nodes_[-1]["lost_inc"] = lost_inc
                super_nodes_[-1]["lost_out"] = lost_out
                
                
                # And finally put x = c
                x = e_start_node
                                       

                inner_iter += 1
                continue


            # Step 7: If the start of edge e is in W then W := W U Vp, F := F U Ep U {e}, and go to step 2
            if e_start_node in W:
                print(f"\t\tThis edge starts in node {e_start_node} which is in W (already added to tree). In this case we're done with inner iterations")
                print(f"\t\tWe flush the path nodes Vp into W and path edges Ep into F and also add edge {e}")
                print(f"\t\tAnd we go all the way back to outer iterations.")
                W = list(set(W).union(set(Vp)))
                F = list(set(F).union(set(Ep)).union(set([e])))
                break
        outer_iter += 1

    print('=====================================================================================================')
    print(f'Main algorithm finished.')
    print(f'W = {W}')
    print(f'F = {F}')
    print(f'all_covered_nodes =')
    print(all_covered_nodes_)

    active_edges = set(F)

    # Step 8: Expand all supernodes starting with the last one that's been added
    # Do so by adding cycle_edges that made the node, except for the last one (marked as non-edge in the dict)
    print(f'Starting supernode expansion')
    for super_node in reversed(super_nodes_):
        node_idx = super_node["node_idx"]
        print(f'\tExpanding supernode {node_idx}')
        lost_inc = super_node["lost_inc"]
        lost_out = super_node["lost_out"]
        cycle_edges = super_node["cycle_edges"]
        non_edge    = super_node["non_edge"]

        active_edges = set(filter(lambda edge: edge[0] != node_idx and edge[1] != node_idx, active_edges))

        for lost_in in lost_inc:
            print(f'\t\tAdding inc {lost_in}')
            active_edges.add(lost_in)
        
        for lost_out in lost_out:
            print(f'\t\tAdding outg {lost_out}')
            active_edges.add(lost_out)

        for cycle_edge in cycle_edges:
            active_edges.add(cycle_edge)

        active_edges = set(filter(lambda edge: edge != non_edge, active_edges))


    result = {
        "active_edges": active_edges,
        "edge_sum": sum(map(lambda x: x[2], active_edges)),
        "root": r
    }
    print(f'Finally, all active edges are: {sorted(list(active_edges))}')
    return result

        
        


    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input',
                        required=True,
                        help='The input file.')

    parser.add_argument('-f',
                        '--format',
                        default='edge_list',
                        help='Input format, either `edge_list` or `matrix`.')

    parser.add_argument('-r',
                        '--root',
                        default='auto',
                        help='The root node.')

    parser.add_argument('-v',
                        '--visualize',
                        help='Visualize the resulting tree.',
                        action='store_true')


    args = parser.parse_args()
    g = DirectedGraph.from_args(args)
    original_edges = list(g.edges())


    result = edmonds(g)
    sys.stdout = STDOUT

    active_edges = list(result["active_edges"])
    print(f'Total edge weight sum: {result["edge_sum"]}')
    print(f'Edges: ')
    for edge in list(sorted(active_edges)):
        print(f'\t{edge}')

    if args.visualize:
        nxg = g.to_networkx_graph()    

        active_edges_nw = list(map(lambda e: (e[0], e[1]), active_edges))
        nx_all_edges = nxg.edges()  
        
        

        edge_colors = ['r' if e in active_edges_nw else 'k' for e in nx_all_edges]
        edge_widths = [3 if e in active_edges_nw else 2 for e in nx_all_edges]
        node_names = dict([(i, f'{i} (root)') if i == result["root"] else (i, str(i)) for i in range(g.num_nodes)])
        node_colors = ['darkorange' if i == result["root"] else 'aqua' for i in range(g.num_nodes)]
        
        pos = nx.circular_layout(nxg)
        edge_labels = nx.get_edge_attributes(nxg, 'weight')       

        nx.draw(nxg, pos, with_labels=True, labels=node_names, node_color=node_colors, edge_color=edge_colors, width=edge_widths)
        nx.draw_networkx_edge_labels(nxg, pos, edge_labels=edge_labels)

        plt.show()