# Revised Simplex Algorithm

Linear programming problem solver using Revised Simplex Algorithm.
__________________________________________
### Usage:

| Flag      | Meaning |
| ----------- | ----------- |
| -h, --help      | Show a help message generated by the `argparse` module.       |
| --format      | Show the input file format.       |
| -i, --input   | Specify the input file.        |
| -e, --eta   | Use ETA matrices.        |
| -m, --max   | Maximize the target function (minimization is default).        |
| -g, --greater   | Use >= instead of <= in constraints (<= is default).        |
| -p, --printproblem   | Print a human readable representation of the problem before starting.|
| -l, --logging   | Print log messages throughout the algorithm.   |

__________________________________________

### Input format:
```
M N
c1 c2 ... cn
a11 a12 ... a1n b1
a21 a22 ... a2n b2
 ... 
am1 am2 ... amn bm
```
where `(aij)` is the `M x N` constraint matrix, `(bi)` is the RHS vector, `(cj)` is the target function. Extra spaces/tabs/line breaks will be ignored.

__________________________________________
### Modules:

- [`numpy`](https://numpy.org/)
- [`portion`](https://pypi.org/project/portion/)



