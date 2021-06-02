# Edmonds algorithm

Finds shortest arborescence of a directed graph.
__________________________________________
### Usage:

| Flag      | Meaning |
| ----------- | ----------- |
| -h, --help      | Show a help message generated by the `argparse` module.       |
| -i, --input   | Specify the input file.        |
| -f, --format   | Input format, either `edge_list` or `matrix`.        |
| -r, --root   | Specify the root node. |
| -v, --visualize   | Visualize the resulting tree. |
| -p, --prog   | Which graphviz program to use for visualization. |


__________________________________________

### Input format 1 (edge list):
```
u0 v0 w0
u1 v1 w1
...
```
where there is an edge of length `w_i` between nodes `u_i` and `v_i`

__________________________________________

### Input format 2 (adjacency matrix):
```
b0_0     b0_1   ... b0_n-1
b1_0     b1_1   ... b1_n-1
...
bn-1_0   bn-1_1 ... bn-1_n-1

```
where `b_i_j` is the edge weight between nodes `i` and `j` (0 means not connected)






