import numpy as np
import portion as P
from scipy.optimize import linprog

class IntegerDomain:

    @staticmethod
    def Binary():
        return IntegerDomain([0, 1])
    

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
                 level=0,
                 parent=None,
                 parent_opt=None):

        self.fixed_vars_mask        = fixed_vars_mask
        self.fixed_vars_values      = fixed_vars_values
        self.next_var               = next_var
        self.next_var_val           = next_var_val
        self.level                  = level
        self.parent                 = parent
        self.parent_opt             = parent_opt

    def __str__(self):
        head = ' -- Search Node --'
        fvars = f'Fixed vars: {self.fixed_vars_mask}'
        fvals = f'Fixed vals: {self.fixed_vars_values}' 
        popt = f'Parent opt: {self.parent_opt}'
        tab = '\t'*self.level
        return '\r\n'.join([tab + head, tab + fvars, tab + fvals, tab + popt])



def mask_and_vals_to_list(mask, vals):
    return [(idx, v) for idx, flag, v in enumerate(zip(mask, vals)) if flag ==1]

def exp_lower_bound(linexp, domains):
    r"""
        Get the lower bound for a linear expression `linexp` where `domains` holds variable domains
    """
    substitutes = np.array([d.upper if c < 0 else d.lower for d, c in zip(domains, linexp)])
    return np.dot(linexp, substitutes)

def exp_lower_bound_fixed(linexp, domains, fixed_vars_mask, fixed_vars_values):
    the_sum = 0
    for fflag, fval, d, coeff in zip(fixed_vars_mask, fixed_vars_values, domains, linexp):
        if fflag == 1:
            the_sum += coeff*fval
        else:
            the_sum += d.upper*coeff if coeff < 0 else d.lower*coeff
    
    return the_sum

def check_constr_lower_bound_fixed(a, b, domains, fixed_vars_mask, fixed_vars_values):
    return exp_lower_bound_fixed(a, domains, fixed_vars_mask, fixed_vars_values) <= b

def check_all_constr_lower_bound_fixed(A, b_vec, domains, fixed_vars_mask, fixed_vars_values):
    vals = [check_constr_lower_bound_fixed(a, b, domains, fixed_vars_mask, fixed_vars_values) for a, b in zip(A, b_vec)]
    return all(vals)


def implicit_enum_method(c, A, b, d):
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

    global_lower_bound = exp_lower_bound(c, d)

    
    bfs_queue       = [BFSData(fixed_vars_mask      = np.zeros(n), 
                               fixed_vars_values    = np.zeros(n),
                               next_var             = -1,
                               next_var_val         = None,
                               level                = 0,
                               parent               = None,
                               parent_opt           = None)]

    while bfs_queue:

        

        current_node_data = bfs_queue.pop(0)

        def printl(*args, **kwargs):
            print('\t'*current_node_data.level, *args, **kwargs)

        if current_node_data.next_var >= n:
            print('\t'*current_node_data.level + 'Leaf node found, this branch is done!')
            continue
        
        printl(f'GLOBAL OPTIMUM: {opt_val}')
        printl(f'Supposed to fix variable x_{current_node_data.next_var} to value {current_node_data.next_var_val}')

        feasible        = True
        maybe_optimal   = True
        new_fixed_vars_mask = current_node_data.fixed_vars_mask.copy()
        new_fixed_vars_values = current_node_data.fixed_vars_values.copy()
        if current_node_data.next_var != -1:
            # Now we fix another variable and we check for feasibility
            new_fixed_vars_mask[current_node_data.next_var] = 1
            new_fixed_vars_values[current_node_data.next_var] = current_node_data.next_var_val
            printl(f'Attempting to fix variable x_{current_node_data.next_var} to value {current_node_data.next_var_val}')

            feasible        = check_all_constr_lower_bound_fixed(A, b, d, new_fixed_vars_mask, new_fixed_vars_values)

            maybe_optimal   = exp_lower_bound_fixed(c, d, new_fixed_vars_mask, new_fixed_vars_values) <= opt_val
        

            if not feasible:
                printl(f'Unfeasible.')
                continue
            
            if not maybe_optimal:
                printl(f'Parent lower bound is {opt_val} but this can only go as low as {exp_lower_bound_fixed(c, d, new_fixed_vars_mask, new_fixed_vars_values)}, so we PRUNE!')
                continue

            current_node_data.fixed_vars_mask = new_fixed_vars_mask
        
        if (current_node_data.fixed_vars_mask == 1).all():
            printl(f'Terminal node found, point {current_node_data.fixed_vars_values}')
            final_val = exp_lower_bound_fixed(c, d, current_node_data.fixed_vars_mask, current_node_data.fixed_vars_values)
            printl(f'THIS POINT GIVES VALUE: {final_val}')
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
                                level=current_node_data.level+1,
                                parent=current_node_data,
                                parent_opt=None) # TODO: Remove parent opt?

            bfs_queue.append(new_node)


    print(f'opt val: {opt_val}')
    print(f'opt_points: ')
    for op in opt_points:
        print(f'\t{op}')
        

        






    



def example1():
    c = [-4, -3, -3, -2]
    A = [[-2, -2, 1, 4],
         [2, 3, -2, 5],
         [1, 4, 5, -1]]
    b = [5, 7, 6]
    d = [IntegerDomain.Binary() for _ in c]

    implicit_enum_method(c, A, b, d)    


if __name__ == '__main__':
    example1()
