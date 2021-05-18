import numpy as np
from scipy.optimize import linprog

if __name__ == '__main__':
    c = [-5.0,  7.0, -9.0,  11.0, 1.0, 1.0,  1.0,  0.0,  0.0,  0.0]
    A = [[1.0, -1.0,  1.0,  -1.0, -1.0, -0.0, -0.0,  1.0,  0.0,  0.0],
         [1.0, -2.0,  3.0,  -4.0, -0.0, -1.0, -0.0,  0.0,  1.0,  0.0],
         [3.0, -4.0,  5.0,  -6.0, -0.0, -0.0, -1.0,  0.0,  0.0,  1.0]]

    b = [10, 6, 15]

    print(linprog(c, A_eq=A, b_eq=b))