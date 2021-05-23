"""
Implements the functionality of a directed graph, and some util functions.
"""

from typing import Iterable
import copy
import sys


def read_lines_ds(filepath):
    try:
        lines = open(filepath, "r").readlines()
        lines = (line.strip() for line in lines)
        lines = (line for line in lines if line)
        lines = (line for line in lines if line[0] != '#')
        lines = list(lines)
        return lines
    except:
        raise ValueError(f'Could not open graph file: {filepath}')
        

class DirectedGraph:
    @staticmethod
    def from_edges(edges: Iterable[tuple(int, int, float)]):
        last_node_idx = max(edges, key=lambda x: max(x[0], x[1]))
        num_nodes = last_node_idx + 1
        g = DirectedGraph(num_nodes)
        for (u, v, w) in edges:
            g.connect(u, v, w)
        return g
    
    @staticmethod
    def from_matrix(matrix: Iterable[Iterable[float]]):
        n = len(matrix)
        if n == 0:
            return DirectedGraph(0)
        if len(matrix[0]) != n:
            raise ValueError(f'Expecting square matrix, got {n}x{len(matrix[0])}')

        g = DirectedGraph(n) 
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != 0:
                    g.connect(i, j, matrix[i][j])
        return g

    @staticmethod
    def from_lines(lines: Iterable[str], format: str):
        format_ = format.lower()
            
        if format_ == 'edge_list':
            edge_list = []
            for line in lines:
                line_strs = line.split(" ")
                if len(line_strs) != 3:
                    raise ValueError(f'Expecting 3 values in edge list, got {line}')
                
                from_str    = line_strs[0]
                to_str      = line_strs[1]
                weight_str  = line_strs[2]

                try:
                    node_from = int(from_str)
                    node_to = int(to_str)
                    edge_weight = float(weight_str)
                    edge_list.append((node_from, node_to, edge_weight))

                except:
                    raise ValueError(f'Expecting (int, int, float) in edge list, got {line}')

            return DirectedGraph.from_edges(edge_list)
        elif format == 'matrix':
            first_row = lines[0]            
            n = len(first_row.split(" "))
            if n == 0:
                return DirectedGraph(0)
            matrix = [[0 for _ in range(n)] for _ in range(n)]
            for i in range(len(lines)):
                line = lines[i]
                line_strs = line.split(" ")
                if len(line_strs) != n:
                    raise ValueError(f'Expected {n} items in row, got {line} with {len(line_strs)} items')

                for j in range(n):
                    entry_str = line_strs[j]
                    try:
                        weight = float(entry_str)
                        matrix[i][j] = weight
                    except:
                        raise ValueError(f'Expected float as weight, got {entry_str}')
                
            return DirectedGraph.from_matrix(matrix)
        else:
            raise ValueError(f'Unknown format: {format}')

    @staticmethod
    def from_file(file_path: str, format: str):
        lines = read_lines_ds(file_path)
        return DirectedGraph.from_lines(lines, format)

    @staticmethod
    def from_args(cmd_args):
        return DirectedGraph.from_file(cmd_args.input, cmd_args.format)

    def __init__(self, num_nodes: int) -> None:
        self.num_nodes          = num_nodes
        self.adj_list           = [[] for _ in range(num_nodes)]
        self.in_degrees         = [0 for _ in range(num_nodes)]
        self.out_degrees        = [0 for _ in range(num_nodes)]

    def connect(self, node_from: int, node_to: int, weight: float) -> None:
        if list(filter(lambda entry: entry[0] == node_to, self.adj_list[node_from], self.adj_list[node_from])):
            raise ValueError(f'Node {node_from} is already connected to node {node_to}.')

        self.adj_list[node_from].append((node_to, weight))
        self.in_degrees[node_to] += 1
        self.out_degrees[node_from] += 1

    def disconnect(self, node_from: int, node_to: int) -> None:
        new_list = list(filter(lambda entry: entry[0] != node_to, self.adj_list[node_from]))
        if len(new_list) == len(self.adj_list[node_from]):
            return
        self.adj_list[node_from] = new_list
        self.out_degrees[node_from] -= 1
        self.in_degrees[node_to] -= 1


def bfs(graph: DirectedGraph, start: int = None) -> dict:
    if start is None:
        start = 0
    if start < 0 or start > graph.num_nodes:
        raise ValueError(f'Start node out of bounds: {start}')


    q           = [start]
    visited     = [False for _ in range(graph.num_nodes)]
    in_numbers  = [-1 for _ in range(graph.num_nodes)]
    nodes_      = []

    visit_id_   = 0

    while q:
        node = q.pop(0)
        if visited[node]:
            continue

        visited[node] = True
        in_numbers[node] = visit_id_
        visit_id_ += 1
        nodes_.append(node)

        for v, _ in graph.adj_list[node]:
            if visited[v]:
                continue
            
            q.append(v)

    total_visited_ = sum(visited)

    return {
        "total_nodes_visited": total_visited_,
        "nodes": nodes_,
        "visited": visited,
        "numbers": in_numbers
    }

    

    


    
                



    

