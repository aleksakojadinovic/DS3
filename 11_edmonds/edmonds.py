import numpy as np
import argparse

from graphs import DirectedGraph
from graphs import bfs

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


    args = parser.parse_args()

    g = DirectedGraph.from_args(args)