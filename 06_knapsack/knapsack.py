import numpy as np
import argparse
import sys

def reconstruct_knapsack(dp_matrix, values, weights):
    m, n = dp_matrix.shape
    
    i, j = m-1, n-1
    total_value = dp_matrix[i][j]
    taken = np.zeros(m)
    while True:
        if i == 0:
            break
        
        current = dp_matrix[i][j]
        upper = dp_matrix[i-1][j]
        if upper == current:
            i -= 1
        else:
            taken[i] += 1
            total_value -= values[i]
            j -= weights[i]

    
    if total_value > 0:
        taken[0] = np.floor(total_value / values[0])

    return taken


def knapsack_in_advance(values, weights, max_weight):
    c = np.array(values)
    a = np.array(weights)

    num_items = len(c)
    
    dp_matrix = np.zeros((num_items, max_weight + 1))
    dp_matrix[0, :] = np.array([c[0]*np.floor(y/a[0]) for y in range(max_weight + 1)])

    for k in range(1, num_items):
        for y in range(max_weight + 1):
            opt1 = dp_matrix[k-1][y]
            opt2 = dp_matrix[k][y - a[k]] + c[k] if y - a[k] >= 0 else float('-inf')
            dp_matrix[k][y] = max(opt1, opt2)
    
    return dp_matrix, dp_matrix[num_items - 1][max_weight], reconstruct_knapsack(dp_matrix, values, weights)

def knapsack_backwards(values, weights, max_weight):
    c = np.array(values)
    a = np.array(weights)

    num_items = len(c)

    dp_matrix = np.zeros((num_items, max_weight + 1))
    dp_matrix[0, :] = np.array([c[0]*np.floor(y/a[0]) for y in range(max_weight + 1)])

    for k in range(1, num_items):
        for y in range(max_weight + 1):
            cands = [c[k]*p + dp_matrix[k-1][y - weights[k]*p] for p in range(int(np.floor(y/weights[k])) + 1)]
            val = 0 if len(cands) == 0 else max(cands)
            dp_matrix[k][y] = val

    return dp_matrix, dp_matrix[num_items - 1][max_weight], reconstruct_knapsack(dp_matrix, values, weights)

def reconstruct_zero_one(dp_matrix, weights):
    res = np.zeros(len(weights))
    i, j = len(dp_matrix)-1, len(dp_matrix[0])-1

    while(i > 0):
        if dp_matrix[i][j] != dp_matrix[i-1][j]:
            res[i-1] += 1
            j -= weights[i-1]
        i -= 1

    # res.reverse()
    return res


def zero_one_knapsack(values, weights, max_weight):
    n = len(values)
    dp_matrix = np.zeros((n+1, max_weight+1))

    for i in range(n + 1):
        for y in range(max_weight + 1):
            if i == 0 or y == 0:
                dp_matrix[i][y] = 0
            elif weights[i - 1] <= y:
                dp_matrix[i][y] = max(values[i - 1] + dp_matrix[i - 1][y - weights[i - 1]], dp_matrix[i - 1][y])
            else:
                dp_matrix[i][y] = dp_matrix[i - 1][y]

    return dp_matrix, dp_matrix[n][max_weight], reconstruct_zero_one(dp_matrix, weights)


def print_simple(dp_matrix, opt, taken):
    print(opt)
    print(taken)
    print(dp_matrix)

def print_human_readable(dp_matrix, opt, taken):
    print(f'Optimal value: {int(opt)}')
    print(f'Items: ')
    items_and_units = filter(lambda entry: entry[1] != 0, enumerate(map(int, taken)))
    items_strings = '\r\n'.join(f"\tTake {unit} unit{'s' if unit > 1 else ' '} of item {item}" for item, unit in items_and_units)
    print(items_strings)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', 
                        '--input',
                        help='The input file',
                        required=True)

    parser.add_argument('-d',
                        '--direction',
                        help='Whether to use `in advance` or `backwards`',
                        default='in_advance')

    parser.add_argument('-w',
                        '--words',
                        help='Print a human-readable output',
                        action='store_true')

    parser.add_argument('-z',
                        '--zeroone',
                        help='Solve 0/1 knapsack',
                        action='store_true')

    args = parser.parse_args()


    if args.direction not in ['in_advance', 'backwards']:
        print(f"Invalid direction: " + args.direction)
        sys.exit(1)
    try:
        input_file = open(args.input, "r")
        lines = input_file.readlines()
        lines = (line.strip() for line in lines)
        lines = (line for line in lines if line)
        lines = (line for line in lines if line[0] != '#')
        lines = list(lines)

        if args.zeroone:
            solve_func = zero_one_knapsack
        elif args.direction == 'in_advance':
            solve_func = knapsack_in_advance
        else:
            solve_func = knapsack_backwards

        print_func = print_human_readable if args.words else print_simple
        total = int(lines[0])
        
        values = list(map(int, lines[1].split(" ")))
        weights = list(map(int, lines[2].split(" ")))
        if len(values) != len(weights):
            print('Different lengths for values and weights!')
            sys.exit(1)
        if len(values) == 0:
            sys.exit(0)

        dp_matrix, opt, taken = solve_func(values, weights, total)
        print_func(dp_matrix, opt, taken)
    except:
        print('Failed to parse input file.')
        sys.exit(1)
    
    

