import numpy as np
import argparse
import lp_input_parser as lpp
import time

class IntegerDomain:
    @staticmethod
    def Binary():
        return IntegerDomain([0, 1])
    
    @staticmethod
    def ZeroToN(N):
        return IntegerDomain(list(range(N + 1)))
    

    def __init__(self, numbers):
        self.numbers = set(numbers)
        self.lower = min(self.numbers)
        self.upper = max(self.numbers)

class BFSData:
    def __init__(self,
                 fixed_vars_mask=None,
                 fixed_vars_values=None,
                 next_var=0,
                 next_var_val=None,
                 level=0):

        self.fixed_vars_mask        = fixed_vars_mask
        self.fixed_vars_values      = fixed_vars_values
        self.next_var               = next_var
        self.next_var_val           = next_var_val
        self.level                  = level


def exp_lower_bound_fixed(linexp, domains, fixed_vars_mask, fixed_vars_values):
    vals = [coeff*fval if fflag == 1 else (coeff*d.upper if coeff < 0 else coeff*d.lower) for fflag, fval, d, coeff in zip(fixed_vars_mask, fixed_vars_values, domains, linexp)]
    return sum(vals)

def check_constr_lower_bound_fixed(a, b, domains, fixed_vars_mask, fixed_vars_values):
    return exp_lower_bound_fixed(a, domains, fixed_vars_mask, fixed_vars_values) <= b

def check_all_constr_lower_bound_fixed(A, b_vec, domains, fixed_vars_mask, fixed_vars_values):
    vals = [check_constr_lower_bound_fixed(a, b, domains, fixed_vars_mask, fixed_vars_values) for a, b in zip(A, b_vec)]
    return all(vals)


def implicit_enum_method(c, A, b, d, log=False):
    r"""
        c       - Objective function c_0, .... c_n-1
        A, b    - Constraints Ax <= b
        d       - Integer domains d_0, ... d_n-1 for each variable

    """


    c = np.array(c)
    A = np.array(A)
    b = np.array(b)
    d = np.array(d)

    n = len(c)



    opt_val         = float('inf')
    opt_points      = []

    
    bfs_queue       = [BFSData(fixed_vars_mask      = np.zeros(n), 
                               fixed_vars_values    = np.zeros(n),
                               next_var             = -1,
                               next_var_val         = None,
                               level                = 0)]

    total_terminals = 0
    total_pruned    = 0
    total_entered   = 0
    total_unfeas    = 0


    start_time = time.time()
    
    while bfs_queue:
        
        current_node_data = bfs_queue.pop(0)

        

        def printl(*args, **kwargs):
            if log:
                print('\t'*current_node_data.level, *args, **kwargs)

        if current_node_data.next_var >= n:
            printl('\t'*current_node_data.level + 'Leaf node found, this branch is done!')
            continue
        
        total_entered += 1

        printl(f'> Current global optimum: {opt_val}')

        feasible        = True
        maybe_optimal   = True
        new_fixed_vars_mask = current_node_data.fixed_vars_mask.copy()
        new_fixed_vars_values = current_node_data.fixed_vars_values.copy()
        if current_node_data.next_var != -1:
            # Now we fix another variable and we check for feasibility
            new_fixed_vars_mask[current_node_data.next_var] = 1
            new_fixed_vars_values[current_node_data.next_var] = current_node_data.next_var_val
            printl(f' > Attempting to fix variable x_{current_node_data.next_var} to value {current_node_data.next_var_val}')
            printl(f'\t - which gives {new_fixed_vars_values}')

            feasible        = check_all_constr_lower_bound_fixed(A, b, d, new_fixed_vars_mask, new_fixed_vars_values)
            my_lb = exp_lower_bound_fixed(c, d, new_fixed_vars_mask, new_fixed_vars_values)
            maybe_optimal   = my_lb <= opt_val
        

            if not feasible:
                printl(f'Unfeasible.')
                total_unfeas += 1
                continue
            
            if not maybe_optimal:
                printl('----PRUNING----')
                printl(f' >> Global lower bound is {opt_val}')
                printl(f' >> This can only go as low as {my_lb}')
                printl('---------------')
                total_pruned += 1
                continue

            current_node_data.fixed_vars_mask = new_fixed_vars_mask
            current_node_data.fixed_vars_values = new_fixed_vars_values
        
        if (current_node_data.fixed_vars_mask == 1).all():
            total_terminals += 1
            printl(f' >> Terminal node found, point {current_node_data.fixed_vars_values}')
            final_val = exp_lower_bound_fixed(c, d, current_node_data.fixed_vars_mask, current_node_data.fixed_vars_values)
            printl(f'\t -- with value: {final_val}')
            if final_val == opt_val:
                opt_points.append(current_node_data.fixed_vars_values)
            elif final_val < opt_val:
                opt_val = final_val
                opt_points = [current_node_data.fixed_vars_values]

            continue

        next_var_idx = current_node_data.next_var + 1
        for next_value in d[next_var_idx].numbers:
            new_node = BFSData(fixed_vars_mask=new_fixed_vars_mask,
                                fixed_vars_values=new_fixed_vars_values,
                                next_var=next_var_idx,
                                next_var_val=next_value,
                                level=current_node_data.level+1)

            bfs_queue.append(new_node)

    end_time = time.time()

    result = dict()
    result['bounded']               = len(opt_points) > 0
    result['message']               = f'Optimal solution found.' if result['bounded'] else f'Function is unbounded.'
    result['opt_val']               = opt_val
    result['opt_points']            = opt_points
    result['n_entered']             = total_entered
    result['n_unfeas']              = total_unfeas
    result['n_pruned']              = total_pruned
    result['n_terminals']           = total_terminals
    result['p_unfeas']              = 0.0 if total_entered == 0 else 100.0 * total_unfeas / total_entered
    result['p_pruned']              = 0.0 if total_entered == 0 else 100.0 * total_pruned / total_entered   

    result['execution_time']        = end_time - start_time

    return result
    

def print_result(result, options):
    if options.logging:
        print('='*100)

    print(f" >> {result['message']} << ")

    if result['bounded']:
        print(f"Optimal function value:")
        print(f"\t{result['opt_val']}")
        print(f"Optimum reached for point{'s' if len(result['opt_points']) > 1 else ''}:")
        for opt_point in result['opt_points']:
            print(f'\t{tuple(map(int, opt_point))}')

    if options.stats:
        print('-'*40)
        print(f"Execution time: {np.round(result['execution_time'], 5)}s")
        print(f"Nodes entered: {result['n_entered']}")
        print(f"Leaves entered: {result['n_terminals']}")
        print(f"Branches pruned (as unfeasible): {result['n_unfeas']} - {np.round(result['p_unfeas'], 2)}%")
        print(f"Branches pruned (as non-optimal): {result['n_pruned']} - {np.round(result['p_pruned'], 2)}%")   
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', 
                        '--input',
                         help='The input file.',
                         required=True)

    parser.add_argument('-s',
                        '--stats',
                        action='store_true',
                        help='Show statistics about algorithm execution.')

    parser.add_argument('-l',
                        '--logging',
                        action='store_true',
                        help='Show log messages throughout the algorithm.')

    parser.add_argument('-m',
                        '--maximize',
                        action='store_true',
                        help='Maximize the objective function (minimization is default).')


    args = parser.parse_args()
    lines = lpp.read_lines_ds(args.input)
    
    m, n = lpp.parse_matrix_dimensions(lines[0])
    c    = lpp.parse_n_floats(n, lines[1])
    A, b = lpp.parse_constraint_matrix(m, n, lines[2:2+m])
    d    = [IntegerDomain.Binary() for _ in range(n)]

    if args.maximize:
        c *= -1

    res  = implicit_enum_method(c, A, b, d, log=args.logging)

    if args.maximize:
        res['opt_val'] *= -1

    print_result(res, args)
