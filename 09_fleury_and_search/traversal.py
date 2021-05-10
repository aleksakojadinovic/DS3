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
                        required=True,
                        default='adj_list',
                        help='Input style, either `adj_list` or `matrix`.')


    args = parser.parse_args()
    g = gp.parse_input(args.input, style=args.style)    
    g.dfs(start_node=0, 
          callback=lambda x, _: print(x, end=' '),
          ordered=False)