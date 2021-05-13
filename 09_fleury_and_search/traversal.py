import argparse

from numpy import recarray
import graph_parser as gp
from simple_graph import SimpleGraph
import copy


def dfs_recursive(graph, node, visited, into):        
        if visited[node]:
            return
        visited[node] = True
        into.append(node)

        for x in sorted(list(graph.adj_list[node])):
            if not visited[x]:
                dfs_recursive(g, x, visited, into)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', 
                        '--input',
                         help='The input file',
                         required=True)
    

    parser.add_argument('-s',
                        '--style',
                        default='adj_list',
                        help='Input style, either `adj_list` or `matrix`.')


    parser.add_argument('-t',
                        '--traversal',
                        default='dfs',
                        help='Which traversal to perform (dfs, bfs or both)')

    parser.add_argument('-r',
                        '--recursive',
                        action='store_true',
                        help='Whether to use recursive dfs')

    args = parser.parse_args()

    if args.traversal not in ['dfs', 'bfs', 'both']:
        print(f'Unknown traversal type: {args.traversal}')

    g = gp.parse_input(args.input, style=args.style)    

    do_dfs = 'dfs' == args.traversal or 'both' == args.traversal
    do_bfs = 'bfs' == args.traversal or 'both' == args.traversal

    dfs_list = []
    bfs_list = []
    if do_dfs:
        if not args.recursive:
            g.dfs(start_node=0, 
                callback=lambda x, _: dfs_list.append(x),
                ordered=True)
        else:
            dfs_recursive(g, 0, [False for _ in range(g.num_nodes)], dfs_list)
    if do_bfs:
        g.bfs(start_node=0, 
            callback=lambda x, _: bfs_list.append(x),
            ordered=True)

    if do_dfs:
        print(f"DFS: {' --> '.join(map(str, dfs_list))}")
    if do_bfs:
        print(f"BFS: {' --> '.join(map(str, bfs_list))}")
