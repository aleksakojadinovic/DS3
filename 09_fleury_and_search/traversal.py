import argparse
import graph_parser as gp
from simple_graph import SimpleGraph


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

    args = parser.parse_args()

    if args.traversal not in ['dfs', 'bfs', 'both']:
        print(f'Unknown traversal type: {args.traversal}')

    g = gp.parse_input(args.input, style=args.style)    

    do_dfs = 'dfs' == args.traversal or 'both' == args.traversal
    do_bfs = 'bfs' == args.traversal or 'both' == args.traversal

    dfs_list = []
    bfs_list = []
    if do_dfs:
        g.dfs(start_node=0, 
            callback=lambda x, _: dfs_list.append(x),
            ordered=True)
    if do_bfs:
        g.bfs(start_node=0, 
            callback=lambda x, _: bfs_list.append(x),
            ordered=True)

    if do_dfs:
        print(f"DFS: {' --> '.join(map(str, dfs_list))}")
    if do_bfs:
        print(f"BFS: {' --> '.join(map(str, bfs_list))}")