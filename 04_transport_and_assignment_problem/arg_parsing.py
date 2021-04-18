import argparse
import sys
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', 
                        '--input',
                         help='The input file',
                         required=True)


    return parser


def read_input():
    parser = get_parser()
    args = parser.parse_args()

    try:
        return np.loadtxt(args.input)
    except:
        return None

    
